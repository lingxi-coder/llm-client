use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::sync::{Notify, Semaphore};

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("attachment operation did not complete")
}

struct Gate {
    entered: AtomicUsize,
    active: AtomicUsize,
    peak: AtomicUsize,
    changed: Notify,
    permits: Semaphore,
}

impl Gate {
    fn new() -> Self {
        Self {
            entered: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            changed: Notify::new(),
            permits: Semaphore::new(0),
        }
    }

    async fn hold(&self) {
        struct Active<'a>(&'a AtomicUsize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        let _active = Active(&self.active);
        self.peak.fetch_max(active, Ordering::SeqCst);
        self.entered.fetch_add(1, Ordering::SeqCst);
        self.changed.notify_one();
        self.permits.acquire().await.unwrap().forget();
    }

    async fn wait_for(&self, count: usize) {
        bounded(async {
            while self.entered.load(Ordering::SeqCst) < count {
                self.changed.notified().await;
            }
        })
        .await
    }
}

#[derive(Default)]
struct Resolver {
    reads: AtomicUsize,
    validations: AtomicUsize,
    reuse: AtomicBool,
    revoked: AtomicBool,
    gate: Option<Arc<Gate>>,
}

#[async_trait]
impl AttachmentResolver for Resolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.revoked.load(Ordering::SeqCst) {
            return Err(LlmError::InvalidRequest {
                message: "attachment missing".into(),
            });
        }
        if let Some(gate) = &self.gate {
            gate.hold().await;
        }
        Ok(Bytes::from_static(b"pdf"))
    }

    async fn validate_content_reuse(&self, _: &AttachmentRef) -> Result<bool, LlmError> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        if self.reuse.load(Ordering::SeqCst) && self.revoked.load(Ordering::SeqCst) {
            return Err(LlmError::PermissionDenied {
                message: "attachment access revoked".into(),
            });
        }
        Ok(self.reuse.load(Ordering::SeqCst))
    }
}

#[derive(Default)]
struct FileTransport {
    uploads: AtomicUsize,
    completion_files: Mutex<Vec<Vec<String>>>,
    gate: Option<Arc<Gate>>,
}

fn response(body: Value) -> StreamResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
        body: Bytes::from(body.to_string()),
    }
    .into()
}

#[async_trait]
impl Transport for FileTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        if request.url.ends_with("/files") {
            self.uploads.fetch_add(1, Ordering::SeqCst);
            let multipart = std::str::from_utf8(&request.body).unwrap();
            let filename = multipart
                .split("filename=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap();
            if let Some(gate) = &self.gate {
                gate.hold().await;
            }
            return Ok(response(
                json!({"id": format!("file-{}", filename.trim_end_matches(".pdf"))}),
            ));
        }
        assert!(request.url.ends_with("/responses"));
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let files = body["input"][0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["file_id"].as_str().unwrap().to_owned())
            .collect();
        self.completion_files.lock().unwrap().push(files);
        Ok(response(json!({
            "status": "completed", "model": "m",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]
        })))
    }
}

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "profile_name": "p", "provider_id": "openai", "base_url": "https://api.openai.com/v1",
        "protocol": "open_ai_responses", "auth": "none",
        "models": [{"display_model": "m", "request_model": "m", "billing_model": "m",
            "metadata": {"inputModalities": ["text", "file"]},
            "capability_support": {"documents": "supported"}}]
    }))
    .unwrap()
}

fn request(ids: &[&str]) -> ChatRequest {
    let content: Vec<Value> = ids
        .iter()
        .map(|id| {
            json!({
                "type": "document", "source": {"type": "attachment", "attachment": {
                    "attachment_id": id, "revision": "1", "filename": format!("{id}.pdf"),
                    "media_type": "application/pdf", "size_bytes": 3
                }}
            })
        })
        .collect();
    serde_json::from_value(
        json!({"model": "m", "messages": [{"role": "user", "content": content}]}),
    )
    .unwrap()
}

fn client(resolver: Arc<dyn AttachmentResolver>, http: Arc<FileTransport>) -> LlmClient {
    let mut builder = LlmClientBuilder::with_transport(http, &[profile()]);
    builder.with_attachment_resolver(resolver);
    builder.with_region(Region::International).build().unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        file_account_scope: Some("account".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn unique_reads_and_uploads_are_bounded_parallel_and_preserve_bindings() {
    let reads = Arc::new(Gate::new());
    let uploads = Arc::new(Gate::new());
    let resolver = Arc::new(Resolver {
        gate: Some(reads.clone()),
        ..Default::default()
    });
    let http = Arc::new(FileTransport {
        gate: Some(uploads.clone()),
        ..Default::default()
    });
    let client = client(resolver.clone(), http.clone());
    let work = tokio::spawn(async move {
        client
            .chat()
            .complete(&request(&["a", "b", "a", "c", "d", "e", "f"]), &options())
            .await
    });
    reads.wait_for(4).await;
    assert_eq!(reads.entered.load(Ordering::SeqCst), 4);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 0);
    reads.permits.add_permits(6);
    uploads.wait_for(4).await;
    assert_eq!(uploads.entered.load(Ordering::SeqCst), 4);
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 6);
    uploads.permits.add_permits(6);
    bounded(work).await.unwrap().unwrap();
    assert_eq!(reads.peak.load(Ordering::SeqCst), 4);
    assert_eq!(uploads.peak.load(Ordering::SeqCst), 4);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 6);
    assert_eq!(
        http.completion_files.lock().unwrap()[0],
        ["file-a", "file-b", "file-a", "file-c", "file-d", "file-e", "file-f"]
    );
}

#[tokio::test]
async fn opt_in_reuse_validates_every_request_and_singleflights_reads() {
    let gate = Arc::new(Gate::new());
    let resolver = Arc::new(Resolver {
        reuse: AtomicBool::new(true),
        gate: Some(gate.clone()),
        ..Default::default()
    });
    let http = Arc::new(FileTransport::default());
    let client = client(resolver.clone(), http.clone());
    let work = (0..8)
        .map(|_| {
            let client = client.clone();
            tokio::spawn(async move {
                client
                    .chat()
                    .complete(&request(&["a", "a"]), &options())
                    .await
            })
        })
        .collect::<Vec<_>>();
    gate.wait_for(1).await;
    // All requests are already scheduled; yielding lets their independent
    // validations run while the one byte read remains gated.
    bounded(async {
        while resolver.validations.load(Ordering::SeqCst) < 8 {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 1);
    gate.permits.add_permits(1);
    for task in work {
        bounded(task).await.unwrap().unwrap();
    }
    assert_eq!(resolver.validations.load(Ordering::SeqCst), 8);
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 1);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 1);
    resolver.revoked.store(true, Ordering::SeqCst);
    assert!(matches!(
        client.chat().complete(&request(&["a"]), &options()).await,
        Err(LlmError::PermissionDenied { .. })
    ));
    assert_eq!(resolver.validations.load(Ordering::SeqCst), 9);
    assert_eq!(http.completion_files.lock().unwrap().len(), 8);
}

#[tokio::test]
async fn disabled_reuse_still_resolves_after_a_warm_cache_hit() {
    let resolver = Arc::new(Resolver {
        reuse: AtomicBool::new(true),
        ..Default::default()
    });
    let http = Arc::new(FileTransport::default());
    let client = client(resolver.clone(), http.clone());
    client
        .chat()
        .complete(&request(&["a"]), &options())
        .await
        .unwrap();
    resolver.reuse.store(false, Ordering::SeqCst);
    client
        .chat()
        .complete(&request(&["a"]), &options())
        .await
        .unwrap();
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 2);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 1);
    resolver.revoked.store(true, Ordering::SeqCst);
    assert!(matches!(
        client.chat().complete(&request(&["a"]), &options()).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 3);
    assert_eq!(http.completion_files.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn metadata_conflicts_fail_before_any_resolver_or_upload() {
    let resolver = Arc::new(Resolver::default());
    let http = Arc::new(FileTransport::default());
    let client = client(resolver.clone(), http.clone());
    let mut req = request(&["a", "a"]);
    if let ContentBlock::Document {
        source: DocumentSource::Attachment { attachment },
        ..
    } = &mut req.messages[0].content[1]
    {
        attachment.filename = "changed.pdf".into();
    }
    assert!(matches!(
        client.chat().complete(&req, &options()).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 0);
    assert_eq!(resolver.validations.load(Ordering::SeqCst), 0);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancellation_drops_all_in_flight_reads_and_starts_no_uploads() {
    let gate = Arc::new(Gate::new());
    let resolver = Arc::new(Resolver {
        gate: Some(gate.clone()),
        ..Default::default()
    });
    let http = Arc::new(FileTransport::default());
    let client = client(resolver, http.clone());
    let work = tokio::spawn(async move {
        client
            .chat()
            .complete(&request(&["a", "b", "c", "d", "e"]), &options())
            .await
    });
    gate.wait_for(4).await;
    work.abort();
    assert!(bounded(work).await.unwrap_err().is_cancelled());
    assert_eq!(gate.active.load(Ordering::SeqCst), 0);
    assert_eq!(gate.entered.load(Ordering::SeqCst), 4);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn request_deadline_cancels_parallel_uploads() {
    let gate = Arc::new(Gate::new());
    let resolver = Arc::new(Resolver::default());
    let http = Arc::new(FileTransport {
        gate: Some(gate.clone()),
        ..Default::default()
    });
    let client = client(resolver, http.clone());
    let work = tokio::spawn(async move {
        client
            .chat()
            .complete(
                &request(&["a", "b", "c", "d", "e"]),
                &RequestOptions {
                    total_timeout: Some(Duration::from_millis(100)),
                    ..options()
                },
            )
            .await
    });
    gate.wait_for(4).await;
    assert!(matches!(
        bounded(work).await.unwrap(),
        Err(LlmError::TransportTimeout { .. })
    ));
    assert_eq!(gate.active.load(Ordering::SeqCst), 0);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 4);
    assert!(http.completion_files.lock().unwrap().is_empty());
}

#[derive(Default)]
struct QwenPartialTransport {
    fail_second: bool,
    metadata_waits: AtomicUsize,
    changed: Notify,
    deleted: Mutex<Vec<String>>,
}

impl QwenPartialTransport {
    async fn wait_for_metadata(&self, count: usize) {
        bounded(async {
            while self.metadata_waits.load(Ordering::SeqCst) < count {
                self.changed.notified().await;
            }
        })
        .await
    }

    async fn wait_for_deletes(&self, count: usize) {
        bounded(async {
            loop {
                let changed = self.changed.notified();
                if self.deleted.lock().unwrap().len() >= count {
                    return;
                }
                changed.await;
            }
        })
        .await
    }
}

#[async_trait]
impl Transport for QwenPartialTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        if request.method == "DELETE" {
            let id = request.url.rsplit('/').next().unwrap().to_owned();
            self.deleted.lock().unwrap().push(id.clone());
            self.changed.notify_one();
            return Ok(response(json!({"id": id, "deleted": true})));
        }
        if request.method == "POST" && request.url.ends_with("/files") {
            let body = std::str::from_utf8(&request.body).unwrap();
            let filename = body
                .split("filename=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap();
            if filename == "b.pdf" && self.fail_second {
                return Err(LlmError::InvalidRequest {
                    message: "second upload rejected".into(),
                });
            }
            return Ok(response(json!({
                "id": format!("file-fe-{}", filename.trim_end_matches(".pdf")),
                "filename": filename, "purpose": "file-extract", "bytes": 3, "status": "uploaded"
            })));
        }
        assert_eq!(request.method, "GET");
        assert!(request.url.contains("/files/file-fe-"));
        self.metadata_waits.fetch_add(1, Ordering::SeqCst);
        self.changed.notify_one();
        futures::future::pending().await
    }
}

fn qwen_client(http: Arc<QwenPartialTransport>) -> LlmClient {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "profile_name": "p", "provider_id": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "protocol": "open_ai_chat", "auth": "none",
        "models": [{"display_model": "qwen-long", "request_model": "qwen-long", "billing_model": "qwen-long",
            "metadata": {"inputModalities": ["text", "file"]}, "capability_support": {"documents": "supported"}}]
    })).unwrap();
    let mut builder = LlmClientBuilder::with_transport(http, &[profile]);
    builder.with_attachment_resolver(Arc::new(Resolver::default()));
    builder.with_region(Region::ChinaMainland).build().unwrap()
}

#[tokio::test(start_paused = true)]
async fn qwen_sibling_upload_failure_cleans_up_a_file_still_waiting_for_readiness() {
    let http = Arc::new(QwenPartialTransport {
        fail_second: true,
        ..Default::default()
    });
    let client = qwen_client(http.clone());
    let mut request = request(&["a", "b"]);
    request.model = "qwen-long".into();
    let result = bounded(client.chat().complete(&request, &options())).await;
    assert!(matches!(result, Err(LlmError::InvalidRequest { .. })));
    assert_eq!(http.metadata_waits.load(Ordering::SeqCst), 1);
    http.wait_for_deletes(1).await;
    assert_eq!(*http.deleted.lock().unwrap(), ["file-fe-a"]);
}

#[tokio::test(start_paused = true)]
async fn qwen_cancellation_cleans_up_all_concurrent_uploaded_files() {
    let http = Arc::new(QwenPartialTransport::default());
    let client = qwen_client(http.clone());
    let work = tokio::spawn(async move {
        let mut request = request(&["a", "b"]);
        request.model = "qwen-long".into();
        client.chat().complete(&request, &options()).await
    });
    http.wait_for_metadata(2).await;
    work.abort();
    assert!(bounded(work).await.unwrap_err().is_cancelled());
    http.wait_for_deletes(2).await;
    let mut deleted = http.deleted.lock().unwrap().clone();
    deleted.sort();
    assert_eq!(deleted, ["file-fe-a", "file-fe-b"]);
}

#[tokio::test]
async fn cancelling_the_shared_byte_read_allows_another_validated_caller_to_retry() {
    let gate = Arc::new(Gate::new());
    let resolver = Arc::new(Resolver {
        reuse: AtomicBool::new(true),
        gate: Some(gate.clone()),
        ..Default::default()
    });
    let http = Arc::new(FileTransport::default());
    let client = client(resolver.clone(), http.clone());
    let first = client.clone();
    let first =
        tokio::spawn(async move { first.chat().complete(&request(&["a"]), &options()).await });
    gate.wait_for(1).await;
    let second =
        tokio::spawn(async move { client.chat().complete(&request(&["a"]), &options()).await });
    bounded(async {
        while resolver.validations.load(Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await;
    first.abort();
    assert!(bounded(first).await.unwrap_err().is_cancelled());
    gate.wait_for(2).await;
    gate.permits.add_permits(1);
    bounded(second).await.unwrap().unwrap();
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 2);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 1);
}

struct DefaultReuseResolver(Resolver);

#[async_trait]
impl AttachmentResolver for DefaultReuseResolver {
    async fn resolve(&self, attachment: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.0.resolve(attachment).await
    }
}

#[tokio::test]
async fn default_resolvers_recheck_missing_content_even_when_remote_file_is_cached() {
    let resolver = Arc::new(DefaultReuseResolver(Resolver::default()));
    let http = Arc::new(FileTransport::default());
    let client = client(resolver.clone(), http.clone());
    for _ in 0..2 {
        client
            .chat()
            .complete(&request(&["a", "a"]), &options())
            .await
            .unwrap();
    }
    assert_eq!(resolver.0.reads.load(Ordering::SeqCst), 2);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 1);
    resolver.0.revoked.store(true, Ordering::SeqCst);
    assert!(matches!(
        client.chat().complete(&request(&["a"]), &options()).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(resolver.0.reads.load(Ordering::SeqCst), 3);
    assert_eq!(http.completion_files.lock().unwrap().len(), 2);
}

struct BackingOwner {
    buffer: Vec<u8>,
    drops: Arc<AtomicUsize>,
}

impl AsRef<[u8]> for BackingOwner {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

impl Drop for BackingOwner {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct SlicedResolver {
    drops: Arc<AtomicUsize>,
    reads: AtomicUsize,
    validations: AtomicUsize,
}

#[async_trait]
impl AttachmentResolver for SlicedResolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        let mut buffer = vec![0; 65 * 1024 * 1024];
        buffer[..3].copy_from_slice(b"pdf");
        Ok(Bytes::from_owner(BackingOwner {
            buffer,
            drops: self.drops.clone(),
        })
        .slice(..3))
    }

    async fn validate_content_reuse(&self, _: &AttachmentRef) -> Result<bool, LlmError> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }
}

#[tokio::test]
async fn review_regression_cached_slices_release_large_backing_buffers() {
    let drops = Arc::new(AtomicUsize::new(0));
    let resolver = Arc::new(SlicedResolver {
        drops: drops.clone(),
        reads: AtomicUsize::new(0),
        validations: AtomicUsize::new(0),
    });
    let http = Arc::new(FileTransport::default());
    let client = client(resolver.clone(), http.clone());
    for _ in 0..2 {
        client
            .chat()
            .complete(&request(&["a"]), &options())
            .await
            .unwrap();
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "cached bytes must not pin the resolver's oversized backing allocation"
        );
    }
    assert_eq!(resolver.reads.load(Ordering::SeqCst), 1);
    assert_eq!(resolver.validations.load(Ordering::SeqCst), 2);
    assert_eq!(http.uploads.load(Ordering::SeqCst), 1);
}
