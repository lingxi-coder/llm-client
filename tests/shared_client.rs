//! Concurrent operations share runtime resources while pinning one configuration revision.
use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Barrier, Notify};

const DEADLINE: Duration = Duration::from_secs(5);
const MODEL: &str = "gpt-5.6";
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct ConfigDir(PathBuf);
impl ConfigDir {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "llm-shared-client-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for ConfigDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn profile(name: &str, endpoint: &str) -> ProviderProfile {
    let mut profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    profile.profile_name = name.into();
    profile.base_url = endpoint.into();
    profile.background = ServiceSetting::Disabled;
    profile.models.retain(|m| m.request_model == MODEL);
    assert_eq!(profile.models.len(), 1);
    profile
}

fn request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({
        "model": model,
        "messages": [{"role":"user","content":[{"type":"text","text":"hello 世界"}]}]
    }))
    .unwrap()
}

fn options(credential: &str) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new(credential.into())),
        ..Default::default()
    }
}

fn authorization(request: &HttpRequest) -> Option<&str> {
    request
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.as_str())
}

fn response() -> StreamResponse {
    HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "id":"resp_complete", "model":MODEL, "status":"completed",
            "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
            "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
        }))
        .unwrap()
        .into(),
    }
    .into()
}

struct ConcurrentTransport {
    requests: Mutex<Vec<HttpRequest>>,
    entered: Barrier,
    release: Barrier,
}

#[async_trait]
impl Transport for ConcurrentTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        self.entered.wait().await;
        self.release.wait().await;
        Ok(response())
    }
}

#[tokio::test]
async fn cloned_clients_send_independent_model_effort_tier_and_credentials_concurrently() {
    tokio::time::timeout(DEADLINE, async {
        let http = Arc::new(ConcurrentTransport {
            requests: Mutex::new(vec![]),
            entered: Barrier::new(3),
            release: Barrier::new(3),
        });
        let mut profile = profile("shared", "https://shared.test/v1");
        let mut second = profile.models[0].clone();
        second.display_model = "second-model".into();
        second.request_model = "second-model".into();
        second.billing_model = "second-model".into();
        second.aliases.clear();
        profile.models.push(second);
        let client = LlmClientBuilder::with_transport(http.clone(), &[profile])
            .with_region(Region::International)
            .build()
            .unwrap();
        let mut tasks = Vec::new();
        for (model, effort, tier, secret) in [
            (MODEL, "high", "fast", "first-secret"),
            ("second-model", "low", "standard", "second-secret"),
        ] {
            let client = client.clone();
            tasks.push(tokio::spawn(async move {
                let request: ChatRequest = serde_json::from_value(json!({
                    "model":model,"messages":[],"thinking":{"effort":effort},"service_tier":tier
                }))
                .unwrap();
                client.chat().complete(&request, &options(secret)).await
            }));
        }
        // Both requests must enter the same transport before either can finish.
        http.entered.wait().await;
        {
            let requests = http.requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            for (model, effort, tier, auth) in [
                (MODEL, "high", "fast", "Bearer first-secret"),
                ("second-model", "low", "default", "Bearer second-secret"),
            ] {
                let (sent, body) = requests
                    .iter()
                    .map(|sent| (sent, serde_json::from_slice::<Value>(&sent.body).unwrap()))
                    .find(|(_, body)| body["model"] == model)
                    .unwrap();
                assert_eq!(body["reasoning"]["effort"], effort);
                assert_eq!(body["service_tier"], tier);
                assert_eq!(authorization(sent), Some(auth));
            }
        }
        http.release.wait().await;
        for task in tasks {
            assert_eq!(task.await.unwrap().unwrap().message.text(), "ok");
        }
    })
    .await
    .expect("shared requests must run concurrently without blocking one another");
}

struct FailoverTransport {
    requests: Mutex<Vec<HttpRequest>>,
    primary_entered: Notify,
    release_primary: Notify,
}

#[async_trait]
impl Transport for FailoverTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let old_primary = request.url.starts_with("https://old-primary.test/");
        self.requests.lock().unwrap().push(request);
        if old_primary {
            self.primary_entered.notify_one();
            self.release_primary.notified().await;
            return Ok(HttpResponse {
                status: 429,
                headers: vec![],
                body: Bytes::from_static(br#"{"error":{"message":"rate limited"}}"#),
            }
            .into());
        }
        Ok(response())
    }
}

#[tokio::test]
async fn in_flight_failover_keeps_original_chain_and_credentials_after_configuration_changes() {
    tokio::time::timeout(DEADLINE, async {
        let http = Arc::new(FailoverTransport {
            requests: Mutex::new(vec![]),
            primary_entered: Notify::new(),
            release_primary: Notify::new(),
        });
        let mut primary = profile("primary", "https://old-primary.test/v1");
        let mut fallback = profile("fallback", "https://old-fallback.test/v1");
        for (profile, order) in [(&mut primary, 0), (&mut fallback, 1)] {
            profile.connection.group = Some("shared".into());
            profile.connection.connection_id = Some(profile.profile_name.clone());
            profile.connection.order = order;
            profile.connection.failover.rate_limit = true;
        }
        let (client, config) =
            LlmClientBuilder::with_transport(http.clone(), &[primary.clone(), fallback])
                .with_region(Region::International)
                .build_managed()
                .unwrap();
        let dir = ConfigDir::new();
        config.set_config_dir(&dir.0).await.unwrap();
        config
            .set_tracked_models("openai", [MODEL.into()])
            .await
            .unwrap();
        let live_service = client.chat();
        let in_flight_client = client.clone();
        let in_flight = tokio::spawn(async move {
            let options = RequestOptions {
                fallback_credentials: BTreeMap::from([(
                    "fallback".into(),
                    Secret::new("old-fallback-secret".into()),
                )]),
                ..options("old-primary-secret")
            };
            in_flight_client
                .chat()
                .complete(&request(MODEL), &options)
                .await
        });
        http.primary_entered.notified().await;

        primary.base_url = "https://new-primary.test/v1".into();
        config.add_provider(primary).await.unwrap();
        config.remove_provider("fallback").await.unwrap();
        // A service captured before publication still reads the new configuration.
        let current = live_service
            .complete(&request(MODEL), &options("new-primary-secret"))
            .await
            .unwrap();
        assert_eq!(current.executed_profile.as_deref(), Some("primary"));
        http.release_primary.notify_one();
        assert_eq!(
            in_flight
                .await
                .unwrap()
                .unwrap()
                .executed_profile
                .as_deref(),
            Some("fallback")
        );
        let requests = http.requests.lock().unwrap();
        let seen: Vec<_> = requests
            .iter()
            .map(|r| (r.url.as_str(), authorization(r)))
            .collect();
        assert_eq!(
            seen,
            vec![
                (
                    "https://old-primary.test/v1/responses",
                    Some("Bearer old-primary-secret")
                ),
                (
                    "https://new-primary.test/v1/responses",
                    Some("Bearer new-primary-secret")
                ),
                (
                    "https://old-fallback.test/v1/responses",
                    Some("Bearer old-fallback-secret")
                ),
            ]
        );
    })
    .await
    .expect("configuration publication and a new request must finish while the old request waits");
}

struct NoHttp;
#[async_trait]
impl Transport for NoHttp {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("snapshot queries must never perform HTTP")
    }
}

#[tokio::test]
async fn fixed_snapshot_preserves_route_price_and_token_estimate_after_updates_and_removal() {
    tokio::time::timeout(DEADLINE, async {
        let mut original = profile("priced", "https://priced.test/v1");
        original.models[0].pricing = Some(TokenPricing {
            input_per_million: Some(2.0),
            output_per_million: Some(4.0),
            ..Default::default()
        });
        original.models[0].info.pricing = original.models[0].pricing.clone();
        let (client, config) =
            LlmClientBuilder::with_transport(Arc::new(NoHttp), std::slice::from_ref(&original))
                .with_region(Region::International)
                .build_managed()
                .unwrap();
        let dir = ConfigDir::new();
        config.set_config_dir(&dir.0).await.unwrap();
        config
            .set_tracked_models("openai", [MODEL.into()])
            .await
            .unwrap();
        let snapshot = client.snapshot();
        let route = snapshot.resolve(MODEL).unwrap();
        let context = PricingContext {
            service_tier: Some(ServiceTier::Standard),
            unix_seconds: Some(1_800_000_000),
            ..Default::default()
        };
        let quote = snapshot.price_quote_for_route(&route, &context).unwrap();
        assert_eq!(quote.rates.as_ref().unwrap().input_per_million, Some(2.0));
        #[cfg(feature = "tokenizer-openai")]
        let tokens = snapshot.estimate_local_tokens(&request(MODEL)).unwrap();

        let mut changed = original;
        changed.models[0]
            .pricing
            .as_mut()
            .unwrap()
            .input_per_million = Some(9.0);
        changed.models[0].info.pricing = changed.models[0].pricing.clone();
        config.add_provider(changed).await.unwrap();
        assert_eq!(
            client
                .price_quote(MODEL, None, &context)
                .unwrap()
                .rates
                .unwrap()
                .input_per_million,
            Some(9.0)
        );
        assert_eq!(snapshot.resolve(MODEL).unwrap(), route);
        assert_eq!(
            snapshot.price_quote_for_route(&route, &context).unwrap(),
            quote
        );
        config.remove_provider("priced").await.unwrap();
        assert!(client.resolve(MODEL).is_err());
        assert!(client.price_quote(MODEL, None, &context).is_err());
        assert!(client.price_quote_for_route(&route, &context).is_err());
        assert_eq!(snapshot.resolve(MODEL).unwrap(), route);
        assert_eq!(
            snapshot.price_quote_for_route(&route, &context).unwrap(),
            quote
        );
        #[cfg(feature = "tokenizer-openai")]
        {
            assert!(tokens.input_tokens > 0);
            assert_eq!(
                snapshot.estimate_local_tokens(&request(MODEL)).unwrap(),
                tokens
            );
            assert!(matches!(
                client.estimate_local_tokens(&request(MODEL)),
                Err(LocalTokenCountError::Resolve(_))
            ));
        }
    })
    .await
    .expect("snapshot queries must remain valid after their live configuration is replaced");
}

struct PausedStreamTransport {
    requests: Mutex<Vec<HttpRequest>>,
    release_body: Arc<Notify>,
}

#[async_trait]
impl Transport for PausedStreamTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        self.requests.lock().unwrap().push(request);
        if body["stream"] != true {
            return Ok(response());
        }
        let first = Bytes::from_static(b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_old_stream\",\"model\":\"gpt-5.6\"}}\n\n");
        let release = self.release_body.clone();
        let remaining = stream::once(async move {
            release.notified().await;
            Ok(Bytes::from_static(concat!(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"old stream\"}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_old_stream\",\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n"
            ).as_bytes()))
        });
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: stream::once(async move { Ok(first) })
                .chain(remaining)
                .boxed(),
        })
    }
}

#[tokio::test]
async fn open_stream_finishes_with_original_decoder_and_continuation_scope_after_publication() {
    tokio::time::timeout(DEADLINE, async {
        let release_body = Arc::new(Notify::new());
        let http = Arc::new(PausedStreamTransport {
            requests: Mutex::new(vec![]),
            release_body: release_body.clone(),
        });
        let mut original = profile("streaming", "https://old-stream.test/v1");
        let (client, config) =
            LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&original))
                .with_region(Region::International)
                .build_managed()
                .unwrap();
        let dir = ConfigDir::new();
        config.set_config_dir(&dir.0).await.unwrap();
        config
            .set_tracked_models("openai", [MODEL.into()])
            .await
            .unwrap();
        let snapshot = client.snapshot();
        let options = RequestOptions {
            account_scope: Some("old-account".into()),
            ..options("old-stream-secret")
        };
        let mut stream = client
            .chat()
            .stream(&request(MODEL), &options)
            .await
            .unwrap();
        loop {
            if matches!(
                stream.next().await.unwrap().unwrap(),
                StreamEvent::Start { .. }
            ) {
                break;
            }
        }
        assert!(stream.continuation().is_none());
        original.base_url = "https://new-stream.test/v1".into();
        original.protocol = ProtocolFamily::OpenAiChat;
        config.add_provider(original.clone()).await.unwrap();
        release_body.notify_one();
        let mut text = String::new();
        let mut ended = false;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::TextDelta { text: delta, .. } => text.push_str(&delta),
                StreamEvent::End { .. } => ended = true,
                _ => {}
            }
        }
        assert_eq!(text, "old stream");
        assert!(ended);
        assert_eq!(stream.executed_profile(), "streaming");
        let reference = stream.continuation().unwrap().clone();
        assert_eq!(reference.response_id.as_str(), "resp_old_stream");
        assert_eq!(reference.account_scope, "old-account");
        assert_eq!(reference.profile_name, "streaming");
        assert_eq!(reference.request_model, MODEL);
        let mut followup = request(MODEL);
        followup.continuation = Some(reference);
        assert!(matches!(
            client.chat().complete(&followup, &options).await,
            Err(LlmError::UnsupportedCapability { .. })
        ));
        // Even restoring the Responses protocol cannot move the old continuation
        // onto the newly published endpoint.
        original.protocol = ProtocolFamily::OpenAiResponses;
        config.add_provider(original).await.unwrap();
        assert!(matches!(
            client.chat().complete(&followup, &options).await,
            Err(LlmError::InvalidRequest { .. })
        ));
        // The continuation still works with the retained endpoint and account scope.
        snapshot.chat().complete(&followup, &options).await.unwrap();
        let requests = http.requests.lock().unwrap();
        assert_eq!(
            requests.len(),
            2,
            "the live route mismatch must fail before HTTP"
        );
        for request in requests.iter() {
            assert_eq!(request.url, "https://old-stream.test/v1/responses");
            assert_eq!(authorization(request), Some("Bearer old-stream-secret"));
        }
        let followup_body: Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert_eq!(followup_body["previous_response_id"], "resp_old_stream");
    })
    .await
    .expect("an open stream must finish against its original decoder and connection scope");
}
