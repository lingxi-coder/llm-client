use async_trait::async_trait;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, ProviderProfile, Region, Secret},
    providers::{
        openai::realtime::OpenAiRealtimeConfig,
        qwen::realtime::{
            QwenRealtimeConfig, QwenRealtimeModel, QwenRealtimeRegion, QwenRealtimeRoute,
            QwenRealtimeScope,
        },
        OpenAiClient, QwenClient,
    },
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeLimits, RealtimeSink, RealtimeTransport,
    },
    HttpRequest, LlmClientBuilder, StreamResponse, Transport,
};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

struct NoHttp;
#[async_trait]
impl Transport for NoHttp {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("realtime clients must use the supplied realtime transport")
    }
}

#[derive(Default)]
struct RecordingRealtime {
    requests: Mutex<Vec<RealtimeConnectRequest>>,
    frames: Arc<Mutex<Vec<RealtimeFrame>>>,
}
struct RecordingSink(Arc<Mutex<Vec<RealtimeFrame>>>);
#[async_trait]
impl RealtimeSink for RecordingSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.0.lock().unwrap().push(frame);
        Ok(())
    }
    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}
#[async_trait]
impl RealtimeTransport for RecordingRealtime {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        self.requests.lock().unwrap().push(request);
        Ok(RealtimeConnection {
            outbound: Box::new(RecordingSink(self.frames.clone())),
            inbound: stream::pending().boxed(),
        })
    }
}
fn profile(provider: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":provider,"profile_name":"selected","protocol":"open_ai_chat",
        "base_url":"https://chat.example.test/v1","auth":"api_key","models":[]
    }))
    .unwrap()
}
fn connection(key: &str) -> RealtimeConnectRequest {
    RealtimeConnectRequest {
        endpoint: "wss://api.openai.com/v1/realtime?model=gpt-realtime".into(),
        headers: vec![("Authorization".into(), format!("Bearer {key}"))],
        max_frame_bytes: 1024 * 1024,
    }
}
struct ConfigDir(std::path::PathBuf);
impl ConfigDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "llm-realtime-clients-{}-{}",
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

#[tokio::test]
async fn realtime_requires_no_chat_model_and_uses_each_call_transport_and_headers() {
    let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("openai")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let openai = client.provider::<OpenAiClient>("selected").unwrap();
    let first = Arc::new(RecordingRealtime::default());
    let second = Arc::new(RecordingRealtime::default());
    let a = openai
        .connect_realtime(
            first.clone(),
            connection("first"),
            "account-a",
            OpenAiRealtimeConfig::default(),
            RealtimeLimits::default(),
        )
        .await
        .unwrap();
    let b = openai
        .connect_realtime(
            second.clone(),
            connection("second"),
            "account-b",
            OpenAiRealtimeConfig::default(),
            RealtimeLimits::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        first.requests.lock().unwrap()[0].headers[0].1,
        "Bearer first"
    );
    assert_eq!(
        second.requests.lock().unwrap()[0].headers[0].1,
        "Bearer second"
    );
    assert_eq!(first.frames.lock().unwrap().len(), 1);
    assert_eq!(second.frames.lock().unwrap().len(), 1);
    drop((a, b));
}

#[tokio::test]
async fn realtime_binding_is_captured_at_connect_and_snapshot_handles_remain_fixed() {
    let (client, config) = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("openai")])
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let directory = ConfigDir::new();
    config.set_config_dir(&directory.0).await.unwrap();
    let live = client.provider::<OpenAiClient>("selected").unwrap();
    let fixed = client
        .snapshot()
        .provider::<OpenAiClient>("selected")
        .unwrap();
    let transport = Arc::new(RecordingRealtime::default());
    let pending = live.connect_realtime(
        transport.clone(),
        connection("key"),
        "account",
        OpenAiRealtimeConfig::default(),
        RealtimeLimits::default(),
    );
    config.add_provider(profile("qwen")).await.unwrap();
    assert!(pending.await.is_err());
    assert!(transport.requests.lock().unwrap().is_empty());
    let session = fixed
        .connect_realtime(
            transport.clone(),
            connection("key"),
            "account",
            OpenAiRealtimeConfig::default(),
            RealtimeLimits::default(),
        )
        .await
        .unwrap();
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
    drop(session);
}

#[tokio::test]
async fn realtime_rejects_another_profile_scope_before_connecting() {
    let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("qwen")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let qwen = client.provider::<QwenClient>("selected").unwrap();
    let transport = Arc::new(RecordingRealtime::default());
    let route = QwenRealtimeRoute::new(QwenRealtimeRegion::Singapore, "workspace").unwrap();
    let scope = QwenRealtimeScope::new("other-profile", "account", &route).unwrap();
    let result = qwen
        .connect_realtime(
            transport.clone(),
            route,
            scope,
            Secret::new("key".into()),
            QwenRealtimeConfig::new(QwenRealtimeModel::Qwen35OmniPlusRealtime),
            RealtimeLimits::default(),
        )
        .await;
    assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
    assert!(transport.requests.lock().unwrap().is_empty());
}
