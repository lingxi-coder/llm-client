use async_trait::async_trait;
use lingxi_llm_client::{
    protocol::{ChatRequest, LlmError, ProviderProfile, Region, Secret},
    providers::{
        qwen::tts::{
            QwenTtsRegion, QwenTtsRequest, QwenTtsScope, QWEN_TTS_SINGAPORE_HTTP_ENDPOINT,
        },
        OpenAiClient, ProviderBindingError, QwenClient,
    },
    HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions, StreamResponse, Transport,
};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
struct RecordingTransport(Mutex<Vec<HttpRequest>>);
#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body = if request.url == QWEN_TTS_SINGAPORE_HTTP_ENDPOINT {
            json!({"request_id":"tts-1","output":{"finish_reason":"stop","audio":{
                "url":"https://audio.example.test/out.wav","id":"audio-1","expires_at":1800000000
            }},"usage":{"characters":5}})
        } else {
            json!({"model":"wire-m","choices":[{"message":{"role":"assistant","content":"ok"},
                "finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
        };
        self.0.lock().unwrap().push(request);
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: body.to_string().into(),
        }
        .into())
    }
}
fn profile(provider: &str, models: bool) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":provider,"profile_name":"selected","protocol":"open_ai_chat",
        "base_url":"https://chat.example.test/v1","auth":"api_key",
        "connection":{"group":"shared-group"},
        "models":if models {json!([{"request_model":"wire-m","display_model":"m","billing_model":"wire-m"}])} else {json!([])}
    })).unwrap()
}
fn options(key: &str) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new(key.to_owned())),
        account_scope: Some("account".into()),
        ..Default::default()
    }
}
fn scope(profile: &str) -> QwenTtsScope {
    QwenTtsScope::new(profile, "account", QwenTtsRegion::Singapore, "workspace").unwrap()
}
struct ConfigDir(std::path::PathBuf);
impl ConfigDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "llm-provider-clients-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for ConfigDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn typed_binding_requires_an_exact_profile_of_the_right_provider_and_region() {
    let transport = Arc::new(RecordingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile("qwen", false)])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert_eq!(
        client
            .provider::<QwenClient>("selected")
            .unwrap()
            .profile_name(),
        "selected"
    );
    assert!(matches!(
        client.provider::<OpenAiClient>("selected"),
        Err(ProviderBindingError::ProviderMismatch { .. })
    ));
    assert!(matches!(
        client.provider::<QwenClient>("shared-group"),
        Err(ProviderBindingError::UnknownProfile { .. })
    ));
    assert!(matches!(
        client.provider::<QwenClient>("missing"),
        Err(ProviderBindingError::UnknownProfile { .. })
    ));
    let mut restricted = profile("qwen", false);
    restricted.regions = vec![Region::ChinaMainland];
    let client = LlmClientBuilder::with_transport(transport.clone(), &[restricted])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(matches!(
        client.provider::<QwenClient>("selected"),
        Err(ProviderBindingError::UnavailableRegion { .. })
    ));
    assert!(transport.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn typed_chat_and_unified_chat_share_the_same_request_execution() {
    let transport = Arc::new(RecordingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile("qwen", true)])
        .with_region(Region::International)
        .build()
        .unwrap();
    let request: ChatRequest = serde_json::from_value(json!({
        "model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]
    }))
    .unwrap();
    let opts = options("key");
    let unified = client
        .chat()
        .complete_in("selected", &request, &opts)
        .await
        .unwrap();
    let typed = client
        .provider::<QwenClient>("selected")
        .unwrap()
        .chat()
        .complete(&request, &opts)
        .await
        .unwrap();
    assert_eq!(unified.message.text(), typed.message.text());
    assert_eq!(unified.executed_profile, typed.executed_profile);
    let sent = transport.0.lock().unwrap();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].method, sent[1].method);
    assert_eq!(sent[0].url, sent[1].url);
    assert_eq!(sent[0].headers, sent[1].headers);
    assert_eq!(sent[0].body, sent[1].body);
}

#[derive(Default)]
struct MixedGroupTransport(Mutex<Vec<String>>);

#[async_trait]
impl Transport for MixedGroupTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.lock().unwrap().push(request.url.clone());
        let fallback = request.url.contains("qwen.example.test");
        let response = if fallback {
            HttpResponse {
                status: 200,
                headers: vec![],
                body: json!({"model":"wire-m","choices":[{"message":{"role":"assistant","content":"qwen"},"finish_reason":"stop"}]}).to_string().into(),
            }
        } else {
            HttpResponse {
                status: 503,
                headers: vec![],
                body: json!({"error":{"message":"unavailable"}})
                    .to_string()
                    .into(),
            }
        };
        Ok(response.into())
    }
}

#[tokio::test]
async fn typed_chat_failover_stays_with_its_provider_in_a_mixed_group() {
    let transport = Arc::new(MixedGroupTransport::default());
    let profiles: Vec<ProviderProfile> = [
        ("primary", "openai", "primary.example.test", "a"),
        ("same-provider", "openai", "backup.example.test", "b"),
        ("other-provider", "qwen", "qwen.example.test", "c"),
    ]
    .into_iter()
    .map(|(name, provider, host, connection)| {
        serde_json::from_value(json!({
            "provider_id":provider, "profile_name":name, "protocol":"open_ai_chat",
            "base_url":format!("https://{host}/v1"), "auth":"none",
            "connection":{"group":"mixed", "connection_id":connection,
                "failover":{"serverError":true}},
            "models":[{"request_model":"wire-m","display_model":"m","billing_model":"wire-m"}]
        }))
        .unwrap()
    })
    .collect();
    let client = LlmClientBuilder::with_transport(transport.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let request: ChatRequest = serde_json::from_value(json!({
        "model":"m", "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]
    }))
    .unwrap();
    let typed = client.provider::<OpenAiClient>("primary").unwrap();
    assert!(typed
        .chat()
        .complete(&request, &RequestOptions::default())
        .await
        .is_err());
    let sent = transport.0.lock().unwrap().clone();
    assert_eq!(
        sent.len(),
        2,
        "same-provider backup should still be attempted"
    );
    assert!(sent.iter().all(|url| !url.contains("qwen.example.test")));

    let unified = client
        .chat()
        .complete_in("primary", &request, &RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(unified.executed_profile.as_deref(), Some("other-provider"));
    assert_eq!(transport.0.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn native_resources_need_no_chat_model_and_keep_credentials_per_operation() {
    let transport = Arc::new(RecordingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile("qwen", false)])
        .with_region(Region::International)
        .build()
        .unwrap();
    let qwen = client.provider::<QwenClient>("selected").unwrap();
    assert!(qwen.tts(scope("other-profile")).is_err());
    let service = qwen.tts(scope("selected")).unwrap();
    let request = QwenTtsRequest::new("hello", "Cherry");
    let first = options("first-key");
    let second = options("second-key");
    let (a, b) = tokio::join!(
        service.synthesize(&request, &first),
        service.synthesize(&request, &second)
    );
    a.unwrap();
    b.unwrap();
    let wrong_account = RequestOptions {
        account_scope: Some("other-account".into()),
        ..first
    };
    assert!(service.synthesize(&request, &wrong_account).await.is_err());
    let sent = transport.0.lock().unwrap();
    assert_eq!(sent.len(), 2);
    assert!(sent
        .iter()
        .all(|request| request.url == QWEN_TTS_SINGAPORE_HTTP_ENDPOINT));
    for (request, key) in sent.iter().zip(["first-key", "second-key"]) {
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                && value == &format!("Bearer {key}")));
    }
}

#[tokio::test]
async fn saved_native_resources_revalidate_live_bindings_but_snapshots_remain_fixed() {
    let transport = Arc::new(RecordingTransport::default());
    let (client, config) =
        LlmClientBuilder::with_transport(transport.clone(), &[profile("qwen", false)])
            .with_region(Region::International)
            .build_managed()
            .unwrap();
    let directory = ConfigDir::new();
    config.set_config_dir(&directory.0).await.unwrap();
    let live = client.provider::<QwenClient>("selected").unwrap();
    let fixed = client
        .snapshot()
        .provider::<QwenClient>("selected")
        .unwrap();
    let live_service = live.tts(scope("selected")).unwrap();
    let fixed_service = fixed.tts(scope("selected")).unwrap();
    let request = QwenTtsRequest::new("hello", "Cherry");
    let opts = options("key");
    let not_started = live_service.synthesize(&request, &opts);
    config.add_provider(profile("openai", false)).await.unwrap();
    assert!(
        not_started.await.is_err(),
        "snapshot must be captured when the operation begins"
    );
    fixed_service.synthesize(&request, &opts).await.unwrap();
    assert_eq!(transport.0.lock().unwrap().len(), 1);
}
