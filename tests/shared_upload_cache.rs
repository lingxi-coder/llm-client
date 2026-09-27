use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::{
    future::Future,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Barrier, Notify};

struct Resolver {
    start: Option<Arc<Barrier>>,
}

#[async_trait]
impl AttachmentResolver for Resolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        if let Some(start) = &self.start {
            start.wait().await;
        }
        Ok(Bytes::from_static(b"pdf"))
    }
}

struct Gate {
    entered: Barrier,
    release: Barrier,
}

impl Gate {
    fn new() -> Self {
        Self {
            entered: Barrier::new(2),
            release: Barrier::new(2),
        }
    }

    async fn hold(&self) {
        self.entered.wait().await;
        self.release.wait().await;
    }
}

#[derive(Default)]
struct FileTransport {
    uploads: AtomicUsize,
    completions: AtomicUsize,
    first_upload: Option<Gate>,
    second_completion_missing: Option<Gate>,
    reuse_file_id: bool,
    completed_files: Mutex<Vec<String>>,
}

impl FileTransport {
    fn upload_count(&self) -> usize {
        self.uploads.load(Ordering::SeqCst)
    }

    fn last_file(&self) -> String {
        self.completed_files.lock().unwrap().last().unwrap().clone()
    }
}

fn response(status: u16, body: Value) -> StreamResponse {
    HttpResponse {
        status,
        headers: vec![("content-type".into(), "application/json".into())],
        body: Bytes::from(body.to_string()),
    }
    .into()
}

#[async_trait]
impl Transport for FileTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        if request.url.ends_with("/files") {
            let sequence = self.uploads.fetch_add(1, Ordering::SeqCst) + 1;
            if sequence == 1 {
                if let Some(gate) = &self.first_upload {
                    gate.hold().await;
                }
            }
            let id = if self.reuse_file_id {
                "file-reused".to_owned()
            } else {
                format!("file-{sequence}")
            };
            return Ok(response(200, json!({"id": id})));
        }
        assert!(request.url.ends_with("/responses"));
        let sequence = self.completions.fetch_add(1, Ordering::SeqCst) + 1;
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let file_id = body["input"][0]["content"][0]["file_id"]
            .as_str()
            .expect("automatic attachment must use the uploaded provider file")
            .to_owned();
        if sequence == 2 {
            if let Some(gate) = &self.second_completion_missing {
                gate.hold().await;
                return Ok(response(
                    404,
                    json!({"error": {
                        "message": format!("File {file_id} not found"),
                        "type": "invalid_request_error"
                    }}),
                ));
            }
        }
        self.completed_files.lock().unwrap().push(file_id);
        Ok(response(
            200,
            json!({
                "status": "completed", "model": "gpt-test",
                "output": [{"type": "message", "content": [{
                    "type": "output_text", "text": "ok"
                }]}]
            }),
        ))
    }
}

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "profile_name": "primary", "provider_id": "openai",
        "base_url": "https://api.openai.com/v1",
        "protocol": "open_ai_responses", "auth": "none",
        "models": [{
            "display_model": "gpt-test", "request_model": "gpt-test",
            "billing_model": "gpt-test",
            "metadata": {"inputModalities": ["text", "file"]},
            "capability_support": {"documents": "supported"}
        }]
    }))
    .unwrap()
}

fn changed_profile() -> ProviderProfile {
    let mut profile = profile();
    // Keep every pre-existing file-cache key field identical. The new
    // connection generation must account for this configuration change.
    profile.extra = json!({"headers": {"x-client-version": "new"}});
    profile
}

fn builder(http: Arc<FileTransport>) -> LlmClientBuilder {
    let mut builder = LlmClientBuilder::with_transport(http, &[profile()]);
    builder.with_attachment_resolver(Arc::new(Resolver { start: None }));
    builder.with_region(Region::International)
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": [{
            "type": "document", "source": {"type": "attachment", "attachment": {
                "attachment_id": "document", "revision": "1",
                "filename": "guide.pdf", "media_type": "application/pdf", "size_bytes": 3
            }}
        }]}]
    }))
    .unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        file_account_scope: Some("account".into()),
        ..Default::default()
    }
}

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("cache operation or test barrier did not complete")
}

struct ConfigDir(PathBuf);

impl ConfigDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "llm-shared-upload-{}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
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
async fn cloned_clients_share_one_in_flight_upload() {
    let http = Arc::new(FileTransport {
        first_upload: Some(Gate::new()),
        ..Default::default()
    });
    let ready = Arc::new(Barrier::new(3));
    let mut builder = builder(http.clone());
    builder.with_attachment_resolver(Arc::new(Resolver {
        start: Some(ready.clone()),
    }));
    let client = builder.build().unwrap();
    let first = client.clone();
    let second = client.clone();
    let first = tokio::spawn(async move { first.chat().complete(&request(), &options()).await });
    let second = tokio::spawn(async move { second.chat().complete(&request(), &options()).await });
    bounded(ready.wait()).await;
    let gate = http.first_upload.as_ref().unwrap();
    bounded(gate.entered.wait()).await;
    assert_eq!(http.upload_count(), 1);
    bounded(gate.release.wait()).await;
    bounded(first).await.unwrap().unwrap();
    bounded(second).await.unwrap().unwrap();
    assert_eq!(http.upload_count(), 1);
    assert_eq!(http.completions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn old_upload_finishing_after_publication_cannot_replace_current_cache() {
    let http = Arc::new(FileTransport {
        first_upload: Some(Gate::new()),
        ..Default::default()
    });
    let (client, config) = builder(http.clone()).build_managed().unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let old = client.snapshot();
    let old_request =
        tokio::spawn(async move { old.chat().complete(&request(), &options()).await });
    let gate = http.first_upload.as_ref().unwrap();
    bounded(gate.entered.wait()).await;

    config.add_provider(changed_profile()).await.unwrap();
    let current = client.snapshot();
    bounded(current.chat().complete(&request(), &options()))
        .await
        .unwrap();
    assert_eq!(http.upload_count(), 2);
    assert_eq!(http.last_file(), "file-2");

    bounded(gate.release.wait()).await;
    bounded(old_request).await.unwrap().unwrap();
    assert_eq!(http.last_file(), "file-1");
    bounded(current.chat().complete(&request(), &options()))
        .await
        .unwrap();
    assert_eq!(http.upload_count(), 2);
    assert_eq!(http.last_file(), "file-2");
}

#[tokio::test]
async fn late_old_file_404_does_not_invalidate_current_generation() {
    let http = Arc::new(FileTransport {
        second_completion_missing: Some(Gate::new()),
        // A provider may reuse its textual ID in a different connection
        // generation; file_id comparison alone cannot provide isolation.
        reuse_file_id: true,
        ..Default::default()
    });
    let (client, config) = builder(http.clone()).build_managed().unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let old = client.snapshot();
    old.chat().complete(&request(), &options()).await.unwrap();
    let old_request =
        tokio::spawn(async move { old.chat().complete(&request(), &options()).await });
    let gate = http.second_completion_missing.as_ref().unwrap();
    bounded(gate.entered.wait()).await;

    config.add_provider(changed_profile()).await.unwrap();
    let current = client.snapshot();
    bounded(current.chat().complete(&request(), &options()))
        .await
        .unwrap();
    assert_eq!(http.upload_count(), 2);
    bounded(gate.release.wait()).await;
    bounded(old_request).await.unwrap().unwrap();
    assert_eq!(http.upload_count(), 3);

    bounded(current.chat().complete(&request(), &options()))
        .await
        .unwrap();
    assert_eq!(http.upload_count(), 3);
}

#[tokio::test]
async fn recreating_a_connection_does_not_reuse_its_previous_generation() {
    let http = Arc::new(FileTransport::default());
    let (client, config) = builder(http.clone()).build_managed().unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let old = client.snapshot();
    old.chat().complete(&request(), &options()).await.unwrap();

    config.remove_provider("primary").await.unwrap();
    config.add_provider(profile()).await.unwrap();
    let current = client.snapshot();
    current
        .chat()
        .complete(&request(), &options())
        .await
        .unwrap();
    assert_eq!(http.upload_count(), 2);
    assert_eq!(http.last_file(), "file-2");

    old.chat().complete(&request(), &options()).await.unwrap();
    assert_eq!(http.upload_count(), 2);
    assert_eq!(http.last_file(), "file-1");
}

#[tokio::test]
async fn switching_config_directories_never_reuses_an_earlier_namespace() {
    let http = Arc::new(FileTransport::default());
    let (client, config) = builder(http.clone()).build_managed().unwrap();
    let first_dir = ConfigDir::new();
    let second_dir = ConfigDir::new();
    for (index, dir) in [&first_dir, &second_dir, &first_dir]
        .into_iter()
        .enumerate()
    {
        config.set_config_dir(&dir.0).await.unwrap();
        client
            .snapshot()
            .chat()
            .complete(&request(), &options())
            .await
            .unwrap();
        assert_eq!(http.upload_count(), index + 1);
        assert_eq!(http.last_file(), format!("file-{}", index + 1));
    }
}

#[tokio::test]
async fn updating_another_profile_preserves_the_unchanged_upload_cache() {
    let http = Arc::new(FileTransport::default());
    let (client, config) = builder(http.clone()).build_managed().unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let mut other = profile();
    other.profile_name = "secondary".into();
    config.add_provider(other.clone()).await.unwrap();

    client
        .chat()
        .complete_in("primary", &request(), &options())
        .await
        .unwrap();
    assert_eq!(http.upload_count(), 1);
    assert_eq!(http.last_file(), "file-1");

    other.extra = json!({"headers": {"x-client-version": "new"}});
    config.add_provider(other.clone()).await.unwrap();
    let current = client.snapshot();
    assert_eq!(current.provider("secondary").unwrap().extra, other.extra);
    current
        .chat()
        .complete_in("primary", &request(), &options())
        .await
        .unwrap();
    assert_eq!(http.upload_count(), 1);
    assert_eq!(http.last_file(), "file-1");
    assert_eq!(http.completions.load(Ordering::SeqCst), 2);
}

const QWEN_OLD_ENDPOINT: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";
const QWEN_NEW_ENDPOINT: &str =
    "https://newworkspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1";

#[derive(Default)]
struct QwenCleanupTransport {
    requests: Mutex<Vec<HttpRequest>>,
    deleted: Notify,
}

impl QwenCleanupTransport {
    async fn wait_for_deletes(&self, expected: usize) {
        loop {
            let deleted = self.deleted.notified();
            if self
                .requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.method == "DELETE")
                .count()
                >= expected
            {
                return;
            }
            deleted.await;
        }
    }
}

#[async_trait]
impl Transport for QwenCleanupTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let file_id = if request.url.starts_with(QWEN_OLD_ENDPOINT) {
            "file-fe-old"
        } else {
            assert!(request.url.starts_with(QWEN_NEW_ENDPOINT));
            "file-fe-new"
        };
        self.requests.lock().unwrap().push(request.clone());
        if request.method == "DELETE" {
            self.deleted.notify_one();
            return Ok(response(200, json!({"id": file_id, "deleted": true})));
        }
        if request.url.ends_with("/chat/completions") {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            assert_eq!(body["stream"], true);
            return Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".into(), "text/event-stream".into())],
                body: Bytes::from_static(
                    b"data: {\"id\":\"chat-qwen\",\"model\":\"qwen-long\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                ),
            }
            .into());
        }
        assert!(request.url.contains("/files"));
        Ok(response(
            200,
            json!({
                "id": file_id, "filename": "guide.pdf", "purpose": "file-extract",
                "bytes": 3, "status": "processed"
            }),
        ))
    }
}

fn qwen_profile(endpoint: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "profile_name": "qwen-account", "provider_id": "qwen",
        "base_url": endpoint, "protocol": "open_ai_chat", "auth": "bearer",
        "models": [{
            "display_model": "qwen-long", "request_model": "qwen-long",
            "billing_model": "qwen-long",
            "metadata": {"inputModalities": ["text", "file"]},
            "capability_support": {"documents": "supported", "streaming": "supported"}
        }]
    }))
    .unwrap()
}

async fn drain_stream(mut stream: ModelStream) {
    let mut ended = false;
    while let Some(event) = stream.next().await {
        ended |= matches!(event.unwrap(), StreamEvent::End { .. });
    }
    assert!(ended);
}

async fn assert_qwen_cleanup_keeps_original_connection(cancel: bool) {
    let http = Arc::new(QwenCleanupTransport::default());
    let mut builder =
        LlmClientBuilder::with_transport(http.clone(), &[qwen_profile(QWEN_OLD_ENDPOINT)]);
    builder.with_attachment_resolver(Arc::new(Resolver { start: None }));
    let (client, config) = builder
        .with_region(Region::ChinaMainland)
        .build_managed()
        .unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let mut request = request();
    request.model = "qwen-long".into();
    let mut options = RequestOptions {
        credential: Some(Secret::new("old-account-key".into())),
        file_account_scope: Some("old-account".into()),
        ..Default::default()
    };
    let old_stream = client.chat().stream(&request, &options).await.unwrap();
    assert!(!http
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|request| request.method == "DELETE"));

    config
        .add_provider(qwen_profile(QWEN_NEW_ENDPOINT))
        .await
        .unwrap();
    options.credential = Some(Secret::new("new-account-key".into()));
    options.file_account_scope = Some("new-account".into());
    let new_stream = client.chat().stream(&request, &options).await.unwrap();
    bounded(drain_stream(new_stream)).await;

    if cancel {
        drop(old_stream);
    } else {
        bounded(drain_stream(old_stream)).await;
    }
    bounded(http.wait_for_deletes(2)).await;
    let requests = http.requests.lock().unwrap();
    let deleted: Vec<_> = requests
        .iter()
        .filter(|request| request.method == "DELETE")
        .collect();
    assert_eq!(deleted.len(), 2);
    for (request, endpoint, file_id, credential) in [
        (
            deleted[0],
            QWEN_NEW_ENDPOINT,
            "file-fe-new",
            "Bearer new-account-key",
        ),
        (
            deleted[1],
            QWEN_OLD_ENDPOINT,
            "file-fe-old",
            "Bearer old-account-key",
        ),
    ] {
        assert_eq!(request.url, format!("{endpoint}/files/{file_id}"));
        assert_eq!(
            request
                .headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.as_str()),
            Some(credential)
        );
    }
}

#[tokio::test(start_paused = true)]
async fn completed_qwen_stream_cleans_up_with_its_original_endpoint_and_credential() {
    assert_qwen_cleanup_keeps_original_connection(false).await;
}

#[tokio::test(start_paused = true)]
async fn cancelled_qwen_stream_cleans_up_with_its_original_endpoint_and_credential() {
    assert_qwen_cleanup_keeps_original_connection(true).await;
}
