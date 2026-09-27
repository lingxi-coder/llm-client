use async_trait::async_trait;
use lingxi_llm_client::{
    batches::BatchError, embeddings::EmbeddingRequest, files::provider_file_endpoint_fingerprint,
    protocol::*, retrieval::RetrievalError, *,
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::sync::Notify;

struct ConfigDir(PathBuf);
impl ConfigDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "llm-shared-services-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for ConfigDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Mock {
    block_first: AtomicBool,
    entered: Notify,
    release: Notify,
    requests: Mutex<Vec<HttpRequest>>,
}
impl Mock {
    fn new(block_first: bool) -> Arc<Self> {
        Arc::new(Self {
            block_first: AtomicBool::new(block_first),
            entered: Notify::new(),
            release: Notify::new(),
            requests: Mutex::new(Vec::new()),
        })
    }
    fn urls(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.url.clone())
            .collect()
    }
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body = if request.url.contains("/embeddings") {
            json!({"data":[{"index":0,"embedding":[1.0,2.0]}]})
        } else if request.url.contains("/images/") {
            json!({"data":[{"b64_json":"AQID"}]})
        } else if request.url.contains("/vector_stores") {
            json!({"id":"vs_1","name":"store","status":"completed"})
        } else if request.url.contains("/batches") {
            let job = json!({"id":"batch_1","endpoint":"/v1/responses", "input_file_id":"file_1","status":"completed"});
            if request.url.contains('?') {
                json!({"data":[job],"has_more":false})
            } else {
                job
            }
        } else {
            panic!("unexpected request: {}", request.url);
        };
        self.requests.lock().unwrap().push(request);
        if self.block_first.swap(false, Ordering::Relaxed) {
            self.entered.notify_one();
            self.release.notified().await;
        }
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: serde_json::to_vec(&body).unwrap().into(),
        }
        .into())
    }
}

fn profile(host: &str) -> ProviderProfile {
    let root = format!("https://{host}.test/v1");
    serde_json::from_value(json!({
        "provider_id":"openai","profile_name":"test","protocol":"open_ai_responses",
        "base_url":root,"auth":"none",
        "embeddings":{"mode":"enabled","value":{
            "api":"open_ai","endpoint":format!("{root}/embeddings"),"auth":{"type":"none"}
        }},
        "retrieval":{"mode":"enabled","value":{
            "api":"open_ai","endpoint":format!("{root}/vector_stores"),"auth":{"type":"none"}
        }},
        "batches":{"mode":"enabled","value":{
            "api":"open_ai","endpoint":format!("{root}/batches"),
            "files_endpoint":format!("{root}/files"),"auth":{"type":"none"}
        }},
        "images":{
            "routes":{"image":{"api":"open_ai","base_url":root}},
            "models":[{"display_model":"image","request_model":"gpt-image-1.5",
                "route":"image","capabilities":{"text_to_image":true}}]
        }
    }))
    .unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        account_scope: Some("account".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn saved_embedding_facade_refreshes_after_commit_while_in_flight_and_fixed_calls_stay_old() {
    let mock = Mock::new(true);
    let (client, config) = LlmClientBuilder::with_transport(mock.clone(), &[profile("old")])
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let snapshot = client.snapshot();
    let service = client.embeddings();
    let request = EmbeddingRequest {
        model: "embedding".into(),
        input: vec!["hello".into()],
        dimensions: None,
        task: None,
    };
    let options = options();
    // Both futures exist before the commit; only the first has started.
    let in_flight = service.embed("test", &request, &options);
    let not_started = service.embed("test", &request, &options);
    let update = async {
        tokio::time::timeout(Duration::from_secs(5), mock.entered.notified())
            .await
            .expect("request should reach the transport");
        tokio::time::timeout(Duration::from_secs(5), config.add_provider(profile("new")))
            .await
            .expect("configuration publication must not wait for network requests")
            .unwrap();
        mock.release.notify_one();
    };
    let (response, ()) = tokio::join!(in_flight, update);
    response.unwrap();
    not_started.await.unwrap();
    snapshot
        .embeddings()
        .embed("test", &request, &options)
        .await
        .unwrap();
    assert_eq!(
        mock.urls(),
        [
            "https://old.test/v1/embeddings",
            "https://new.test/v1/embeddings",
            "https://old.test/v1/embeddings",
        ]
    );
}

#[tokio::test]
async fn saved_image_facade_refreshes_after_commit_and_fixed_snapshot_keeps_old_route() {
    let mock = Mock::new(true);
    let (client, config) = LlmClientBuilder::with_transport(mock.clone(), &[profile("old")])
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let snapshot = client.snapshot();
    let service = client.images();
    let request = ImageGenerationRequest {
        model: "image".into(),
        prompt: "blue cat".into(),
        references: vec![],
        output: Default::default(),
        provider_options: Default::default(),
    };
    let options = ImageRequestOptions {
        credential: Some(Secret::new("test-key".into())),
        ..Default::default()
    };
    let update = async {
        tokio::time::timeout(Duration::from_secs(5), mock.entered.notified())
            .await
            .expect("request should reach the transport");
        tokio::time::timeout(Duration::from_secs(5), config.add_provider(profile("new")))
            .await
            .unwrap()
            .unwrap();
        mock.release.notify_one();
    };
    let (response, ()) = tokio::join!(service.generate(&request, &options), update);
    response.unwrap();
    service.generate(&request, &options).await.unwrap();
    snapshot
        .images()
        .generate(&request, &options)
        .await
        .unwrap();
    assert_eq!(
        mock.urls(),
        [
            "https://old.test/v1/images/generations",
            "https://new.test/v1/images/generations",
            "https://old.test/v1/images/generations",
        ]
    );
}

#[tokio::test]
async fn retrieval_and_batch_references_stay_scoped_to_their_snapshot_after_endpoint_change() {
    let mock = Mock::new(false);
    let (client, config) = LlmClientBuilder::with_transport(mock.clone(), &[profile("old")])
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let dir = ConfigDir::new();
    config.set_config_dir(&dir.0).await.unwrap();
    let snapshot = client.snapshot();
    let retrieval = client.retrieval();
    let batches = client.batches();
    let options = options();
    let store = retrieval
        .create_store("test", "store", &options)
        .await
        .unwrap();
    let jobs = batches.list("test", 1, None, &options).await.unwrap();
    let job = &jobs.jobs[0].reference;

    config.add_provider(profile("new")).await.unwrap();
    let sent = mock.urls().len();
    assert!(matches!(
        retrieval.get_store(&store.reference, &options).await,
        Err(RetrievalError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(matches!(
        batches.get(job, &options).await,
        Err(BatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert_eq!(
        mock.urls().len(),
        sent,
        "scope mismatches must fail before dispatch"
    );
    snapshot
        .retrieval()
        .get_store(&store.reference, &options)
        .await
        .unwrap();
    snapshot.batches().get(job, &options).await.unwrap();
    let current_store = retrieval
        .create_store("test", "store", &options)
        .await
        .unwrap();
    let current_jobs = batches.list("test", 1, None, &options).await.unwrap();
    assert_eq!(
        current_store.reference.endpoint_fingerprint,
        provider_file_endpoint_fingerprint("https://new.test/v1/vector_stores")
    );
    assert_eq!(
        current_jobs.jobs[0].reference.endpoint_fingerprint,
        provider_file_endpoint_fingerprint("https://new.test/v1/batches")
    );
    let urls = mock.urls();
    assert!(urls[2].starts_with("https://old.test/"));
    assert!(urls[3].starts_with("https://old.test/"));
    assert!(urls[4].starts_with("https://new.test/"));
    assert!(urls[5].starts_with("https://new.test/"));
}
