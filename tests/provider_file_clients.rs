use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use lingxi_llm_client::{
    files::{
        provider_file_endpoint_fingerprint, FilePurpose, ProviderFileRef, UploadFile,
        UploadFileStream,
    },
    protocol::{AuthStrategy, LlmError, ProtocolFamily, ProviderProfile, Region, Secret},
    providers::{AnthropicClient, GoogleClient, OpenAiClient, OpenRouterClient, XaiClient},
    Authenticator, HttpRequest, HttpResponse, HttpStreamRequest, LlmClientBuilder, RequestOptions,
    StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

struct RecordingTransport {
    requests: Mutex<Vec<HttpRequest>>,
    responses: Mutex<VecDeque<HttpResponse>>,
}
impl RecordingTransport {
    fn new(responses: impl IntoIterator<Item = HttpResponse>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            responses: Mutex::new(responses.into_iter().collect()),
        }
    }
}
#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("one response per operation")
            .into())
    }
    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        let mut body = BytesMut::new();
        while let Some(chunk) = request.body.next().await {
            body.extend_from_slice(&chunk?);
        }
        assert_eq!(body.len() as u64, request.content_length);
        self.send(HttpRequest {
            http1_header_layout: None,
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: body.freeze(),
            timeout: request.timeout,
        })
        .await
    }
}
struct CustomAuth;
#[async_trait]
impl Authenticator for CustomAuth {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        _profile: &ProviderProfile,
        credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        request.headers.push((
            "x-injected-auth".into(),
            credential.unwrap().expose_secret().clone(),
        ));
        Ok(())
    }
}
struct FailingAuth;
#[async_trait]
impl Authenticator for FailingAuth {
    async fn apply(
        &self,
        _request: &mut HttpRequest,
        _profile: &ProviderProfile,
        _credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        Err(LlmError::Transport {
            message: "credential refresh failed before send".into(),
        })
    }
}
fn response(body: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![],
        body: body.to_string().into(),
    }
}
fn profile(provider: &str) -> ProviderProfile {
    let (endpoint, protocol) = match provider {
        "anthropic" => (
            "https://api.anthropic.com",
            ProtocolFamily::AnthropicMessages,
        ),
        "google" => (
            "https://generativelanguage.googleapis.com/v1beta",
            ProtocolFamily::GeminiGenerateContent,
        ),
        "xai" => ("https://api.x.ai/v1", ProtocolFamily::OpenAiResponses),
        "openrouter" => ("https://openrouter.ai/api/v1", ProtocolFamily::OpenAiChat),
        _ => ("https://api.openai.com/v1", ProtocolFamily::OpenAiResponses),
    };
    serde_json::from_value(json!({"provider_id":provider,"profile_name":"selected","protocol":protocol,"base_url":endpoint,"auth":"api_key","models":[]})).unwrap()
}
fn client(provider: &str, transport: Arc<RecordingTransport>) -> lingxi_llm_client::LlmClient {
    let mut builder = LlmClientBuilder::with_transport(transport, &[profile(provider)])
        .with_region(Region::International);
    builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(CustomAuth));
    builder.build().unwrap()
}
fn options(key: &str, account: &str) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new(key.to_owned())),
        account_scope: Some(account.into()),
        ..Default::default()
    }
}
fn file_ref(provider: &str, file_id: &str, account: &str) -> ProviderFileRef {
    let profile = profile(provider);
    ProviderFileRef {
        provider_id: profile.provider_id,
        profile_name: profile.profile_name,
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
        account_scope: Some(account.into()),
        protocol: profile.protocol,
        file_id: file_id.into(),
        uri: None,
        filename: None,
        media_type: None,
        size_bytes: None,
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: None,
    }
}

#[tokio::test]
async fn ordinary_file_listing_is_available_for_all_native_listing_providers() {
    macro_rules! verify {
        ($provider:literal, $kind:ty, $body:expr, $query:literal) => {{
            let transport = Arc::new(RecordingTransport::new([response($body)]));
            let client = client($provider, transport.clone());
            let provider = client.provider::<$kind>("selected").unwrap();
            assert!(provider
                .files()
                .list(Some("next token"), &options("key", "account"))
                .await
                .unwrap()
                .files
                .is_empty());
            let requests = transport.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert!(requests[0].url.contains($query));
            assert!(requests[0]
                .headers
                .iter()
                .any(|(name, value)| name == "x-injected-auth" && value == "key"));
        }};
    }
    verify!(
        "anthropic",
        AnthropicClient,
        json!({"data":[]}),
        "page=next%20token"
    );
    verify!(
        "google",
        GoogleClient,
        json!({"files":[]}),
        "pageToken=next%20token"
    );
    verify!(
        "xai",
        XaiClient,
        json!({"data":[]}),
        "pagination_token=next%20token"
    );
    verify!(
        "openrouter",
        OpenRouterClient,
        json!({"data":[]}),
        "cursor=next%20token"
    );
}

#[tokio::test]
async fn anthropic_metadata_batch_rejects_cross_account_refs_before_network() {
    let transport = Arc::new(RecordingTransport::new([response(
        json!({"data":[{"id":"file-1"}]}),
    )]));
    let client = client("anthropic", transport.clone());
    let anthropic = client.provider::<AnthropicClient>("selected").unwrap();
    let file = file_ref("anthropic", "file-1", "account");
    assert!(anthropic
        .files()
        .list_by_ids(
            std::slice::from_ref(&file),
            &options("key", "other-account")
        )
        .await
        .is_err());
    assert!(transport.requests.lock().unwrap().is_empty());
    let page = anthropic
        .files()
        .list_by_ids(&[file], &options("key", "account"))
        .await
        .unwrap();
    assert_eq!(page.files[0].file.file_id, "file-1");
    assert!(transport.requests.lock().unwrap()[0]
        .url
        .contains("ids%5B%5D=file-1"));
}

#[tokio::test]
async fn google_split_phase_upload_poll_and_resume_keep_credentials_and_scope_explicit() {
    let pending = json!({"name":"files/abc","uri":"https://generativelanguage.googleapis.com/v1beta/files/abc","mimeType":"text/plain","state":"PROCESSING"});
    let active = json!({"name":"files/abc","uri":"https://generativelanguage.googleapis.com/v1beta/files/abc","mimeType":"text/plain","state":"ACTIVE"});
    let mut start = response(json!({}));
    start.headers.push((
        "x-goog-upload-url".into(),
        "https://generativelanguage.googleapis.com/upload/session-1".into(),
    ));
    let transport = Arc::new(RecordingTransport::new([
        start,
        response(json!({"file":pending})),
        response(active.clone()),
        response(active),
    ]));
    let client = client("google", transport.clone());
    let google = client.provider::<GoogleClient>("selected").unwrap();
    let files = google
        .files()
        .with_upload_timeout(Duration::from_secs(10))
        .with_processing_timeout(Duration::from_secs(10));
    let file = UploadFile {
        filename: "note.txt".into(),
        media_type: "text/plain".into(),
        bytes: Bytes::from_static(b"hello"),
    };
    assert_eq!(
        files
            .upload_unpolled(&file, &options("upload-key", "account"))
            .await
            .unwrap()
            .state,
        "PROCESSING"
    );
    assert_eq!(
        files
            .poll_active(
                "files/abc",
                Duration::from_millis(1),
                Duration::from_secs(1),
                &options("poll-key", "account")
            )
            .await
            .unwrap()
            .state,
        "ACTIVE"
    );
    let mut pending = file_ref("google", "files/abc", "account");
    pending.processing_status = Some("PROCESSING".into());
    pending.uri = Some("https://generativelanguage.googleapis.com/v1beta/files/abc".into());
    assert!(files
        .resume_processing(&pending.model_reference(), &options("key", "other-account"))
        .await
        .is_err());
    assert_eq!(
        files
            .resume_processing(
                &pending.model_reference(),
                &options("resume-key", "account")
            )
            .await
            .unwrap()
            .processing_status
            .as_deref(),
        Some("ACTIVE")
    );
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert!(!requests[1]
        .headers
        .iter()
        .any(|(name, _)| name == "x-injected-auth"));
    for (index, key) in [(0, "upload-key"), (2, "poll-key"), (3, "resume-key")] {
        assert!(requests[index]
            .headers
            .iter()
            .any(|(name, value)| name == "x-injected-auth" && value == key));
    }
}

#[tokio::test]
async fn openai_batch_upload_streams_jsonl_and_purpose_listing_is_separate() {
    let transport = Arc::new(RecordingTransport::new([
        response(json!({"id":"file-batch","purpose":"batch"})),
        response(json!({"data":[]})),
    ]));
    let client = client("openai", transport.clone());
    let openai = client.provider::<OpenAiClient>("selected").unwrap();
    let file = UploadFileStream::from_bytes(
        "requests.jsonl",
        "application/jsonl",
        Bytes::from_static(b"{}\n"),
    );
    let uploaded = openai
        .files()
        .upload_batch_stream(file, &options("key", "account"))
        .await
        .unwrap();
    assert_eq!(uploaded.purpose.as_deref(), Some("batch"));
    openai
        .files()
        .list_for_purpose(FilePurpose::Batch, None, &options("key", "account"))
        .await
        .unwrap();
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("requests.jsonl"));
    assert!(body.contains("\r\n{}\n\r\n"));
    assert!(requests[1].url.contains("purpose=batch"));
}

struct ConfigDir(std::path::PathBuf);
impl ConfigDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "llm-files-clients-{}-{}",
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
async fn file_operations_revalidate_live_clients_while_snapshots_keep_their_binding() {
    let transport = Arc::new(RecordingTransport::new([response(json!({"data":[]}))]));
    let (client, config) =
        LlmClientBuilder::with_transport(transport.clone(), &[profile("openai")])
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
    let files = live.files();
    let options = options("key", "account");
    let pending = files.list(None, &options);
    config.add_provider(profile("anthropic")).await.unwrap();
    assert!(pending.await.is_err());
    assert!(transport.requests.lock().unwrap().is_empty());
    fixed.files().list(None, &options).await.unwrap();
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

struct HangingUpload(AtomicUsize);
#[async_trait]
impl Transport for HangingUpload {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("a file stream must use streaming upload transport")
    }
    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        futures::future::pending().await
    }
}

struct HangingBufferedUpload(AtomicUsize);

#[async_trait]
impl Transport for HangingBufferedUpload {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        futures::future::pending().await
    }
}

#[tokio::test]
async fn buffered_upload_timeout_reports_unknown_outcome_after_dispatch() {
    let transport = Arc::new(HangingBufferedUpload(AtomicUsize::new(0)));
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile("openai")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let openai = client.provider::<OpenAiClient>("selected").unwrap();
    let mut options = options("key", "account");
    options.total_timeout = Some(Duration::from_millis(25));
    let file = UploadFile {
        filename: "note.txt".into(),
        media_type: "text/plain".into(),
        bytes: Bytes::from_static(b"hello"),
    };
    let result = openai
        .files()
        .upload(&file, FilePurpose::ModelInput, &options)
        .await;
    assert!(
        matches!(result, Err(LlmError::FileUploadOutcomeUnknown { .. })),
        "{result:?}"
    );
    assert_eq!(transport.0.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn buffered_upload_success_without_a_usable_file_id_is_outcome_unknown() {
    for (status, body) in [
        (200, Bytes::from_static(b"not-json")),
        (200, Bytes::from_static(b"{}")),
        (503, Bytes::from_static(b"provider error")),
    ] {
        let transport = Arc::new(RecordingTransport::new([HttpResponse {
            status,
            headers: vec![],
            body,
        }]));
        let client = client("openai", transport.clone());
        let openai = client.provider::<OpenAiClient>("selected").unwrap();
        let file = UploadFile {
            filename: "note.txt".into(),
            media_type: "text/plain".into(),
            bytes: Bytes::from_static(b"hello"),
        };
        assert!(matches!(
            openai
                .files()
                .upload(&file, FilePurpose::ModelInput, &options("key", "account"))
                .await,
            Err(LlmError::FileUploadOutcomeUnknown { .. })
        ));
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn gemini_buffered_upload_malformed_final_response_is_outcome_unknown() {
    let transport = Arc::new(RecordingTransport::new([
        HttpResponse {
            status: 200,
            headers: vec![(
                "x-goog-upload-url".into(),
                "https://generativelanguage.googleapis.com/upload/session".into(),
            )],
            body: Bytes::new(),
        },
        HttpResponse {
            status: 200,
            headers: vec![],
            body: Bytes::from_static(b"{}"),
        },
    ]));
    let client = client("google", transport.clone());
    let google = client.provider::<GoogleClient>("selected").unwrap();
    let file = UploadFile {
        filename: "note.txt".into(),
        media_type: "text/plain".into(),
        bytes: Bytes::from_static(b"hello"),
    };
    assert!(matches!(
        google
            .files()
            .upload(&file, FilePurpose::ModelInput, &options("key", "account"))
            .await,
        Err(LlmError::FileUploadOutcomeUnknown { .. })
    ));
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn typed_batch_services_use_the_registered_authenticator() {
    let unauthorized = || HttpResponse {
        status: 401,
        headers: vec![],
        body: Bytes::from_static(b"{}"),
    };
    let anthropic_transport = Arc::new(RecordingTransport::new([unauthorized()]));
    let anthropic = client("anthropic", anthropic_transport.clone())
        .provider::<AnthropicClient>("selected")
        .unwrap();
    let anthropic_scope = lingxi_llm_client::providers::anthropic::batch::AnthropicBatchScope::new(
        "selected",
        "account",
        "https://api.anthropic.com",
        None,
    )
    .unwrap();
    let batch = anthropic.batch(anthropic_scope).unwrap();
    let _ = batch
        .list(
            &lingxi_llm_client::providers::anthropic::batch::AnthropicBatchListOptions::default(),
            &options("anthropic-key", "account"),
        )
        .await;
    {
        let sent = anthropic_transport.requests.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0]
            .headers
            .iter()
            .any(|(name, value)| name == "x-injected-auth" && value == "anthropic-key"));
        assert!(!sent[0]
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("x-api-key")));
    }

    let google_transport = Arc::new(RecordingTransport::new([unauthorized(), unauthorized()]));
    let google = client("google", google_transport.clone())
        .provider::<GoogleClient>("selected")
        .unwrap();
    let google_scope = lingxi_llm_client::providers::google::batch::GeminiBatchScope::new(
        "selected",
        "account",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    .unwrap();
    let file = lingxi_llm_client::providers::google::batch::GeminiBatchFileRef::from_resource_name(
        &google_scope,
        "files/input-123",
    )
    .unwrap();
    let batch = google.batch(google_scope).unwrap();
    let credential = Secret::new("google-token".to_owned());
    let _ = batch
        .list(
            &lingxi_llm_client::providers::google::batch::GeminiBatchListOptions::default(),
            &credential,
        )
        .await;
    let _ = batch.download_results(&file, &credential).await;
    let sent = google_transport.requests.lock().unwrap();
    assert_eq!(sent.len(), 2);
    for request in sent.iter() {
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "x-injected-auth" && value == "google-token"));
        assert!(!request
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key")));
    }
}

#[tokio::test]
async fn gemini_batch_authentication_failure_is_not_an_unknown_upload_outcome() {
    let transport = Arc::new(RecordingTransport::new([]));
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[profile("google")])
        .with_region(Region::International);
    builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(FailingAuth));
    let client = builder.build().unwrap();
    let google = client.provider::<GoogleClient>("selected").unwrap();
    let scope = lingxi_llm_client::providers::google::batch::GeminiBatchScope::new(
        "selected",
        "account",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    .unwrap();
    let batch = google.batch(scope).unwrap();
    let body = futures::stream::once(async { Ok(Bytes::from_static(b"{}\n")) }).boxed();
    let result = batch
        .upload_input_stream(
            "input.jsonl",
            3,
            body,
            &Secret::new("credential".to_owned()),
        )
        .await;
    assert!(matches!(
        result,
        Err(
            lingxi_llm_client::providers::google::batch::GeminiBatchError::Llm(
                LlmError::Transport { .. }
            )
        )
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}

struct ProcessingGeminiUpload(AtomicUsize);

#[async_trait]
impl Transport for ProcessingGeminiUpload {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        if request.url.ends_with("/upload/v1beta/files") {
            return Ok(HttpResponse {
                status: 200,
                headers: vec![(
                    "x-goog-upload-url".into(),
                    "https://generativelanguage.googleapis.com/upload/session".into(),
                )],
                body: Bytes::new(),
            }
            .into());
        }
        if request.url.ends_with("/upload/session") {
            return Ok(response(json!({"file": {
                "name":"files/pending", "state":"PROCESSING",
                "uri":"https://generativelanguage.googleapis.com/v1beta/files/pending"
            }}))
            .into());
        }
        futures::future::pending().await
    }
}

#[tokio::test]
async fn gemini_processing_timeout_keeps_the_resumable_file_reference() {
    let transport = Arc::new(ProcessingGeminiUpload(AtomicUsize::new(0)));
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile("google")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let google = client.provider::<GoogleClient>("selected").unwrap();
    let mut options = options("key", "account");
    options.total_timeout = Some(Duration::from_millis(50));
    let file = UploadFile {
        filename: "note.txt".into(),
        media_type: "text/plain".into(),
        bytes: Bytes::from_static(b"hello"),
    };
    let result = google
        .files()
        .upload(&file, FilePurpose::ModelInput, &options)
        .await;
    match result {
        Err(LlmError::ProviderFileProcessing { file, .. }) => {
            assert_eq!(file.file_id, "files/pending");
        }
        other => panic!("expected resumable file reference, got {other:?}"),
    }
    assert_eq!(transport.0.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn gemini_resume_timeout_keeps_the_resumable_file_reference() {
    let transport = Arc::new(ProcessingGeminiUpload(AtomicUsize::new(0)));
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile("google")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let google = client.provider::<GoogleClient>("selected").unwrap();
    let mut pending = file_ref("google", "files/pending", "account");
    pending.processing_status = Some("PROCESSING".into());
    let mut opts = options("key", "account");
    opts.total_timeout = Some(Duration::from_millis(50));
    let result = google
        .files()
        .resume_processing(&pending.model_reference(), &opts)
        .await;
    match result {
        Err(LlmError::ProviderFileProcessing { file, .. }) => {
            assert_eq!(file.file_id, "files/pending");
        }
        other => panic!("expected resumable file reference, got {other:?}"),
    }
    assert_eq!(transport.0.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn streaming_deadline_keeps_unknown_outcome_after_dispatch_without_retry() {
    let transport = Arc::new(HangingUpload(AtomicUsize::new(0)));
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile("openai")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let openai = client.provider::<OpenAiClient>("selected").unwrap();
    let mut options = options("key", "account");
    options.total_timeout = Some(Duration::ZERO);
    let result = openai
        .files()
        .upload_stream(
            UploadFileStream::from_bytes("note.txt", "text/plain", Bytes::from_static(b"hello")),
            FilePurpose::ModelInput,
            &options,
        )
        .await;
    assert!(matches!(
        result,
        Err(lingxi_llm_client::files::FileUploadError::Llm(
            LlmError::TransportTimeout { .. }
        ))
    ));
    assert_eq!(transport.0.load(Ordering::Relaxed), 0);
    options.total_timeout = Some(Duration::from_millis(5));
    let result = openai
        .files()
        .upload_stream(
            UploadFileStream::from_bytes("note.txt", "text/plain", Bytes::from_static(b"hello")),
            FilePurpose::ModelInput,
            &options,
        )
        .await;
    assert!(
        matches!(result, Err(lingxi_llm_client::files::FileUploadError::OutcomeUnknown { source, .. }) if matches!(*source, LlmError::TransportTimeout { .. }))
    );
    assert_eq!(transport.0.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn request_local_auth_overrides_registry_for_file_list_and_upload() {
    struct RequestAuth;
    #[async_trait]
    impl Authenticator for RequestAuth {
        async fn apply(
            &self,
            request: &mut HttpRequest,
            _: &ProviderProfile,
            _: Option<&Secret<String>>,
        ) -> Result<(), LlmError> {
            request
                .headers
                .push(("authorization".into(), "Bearer request-token".into()));
            Ok(())
        }
    }
    let transport = Arc::new(RecordingTransport::new([
        response(json!({"data":[]})),
        response(
            json!({"id":"file-1","filename":"note.txt","mime_type":"text/plain","size_bytes":5,"type":"file"}),
        ),
        response(json!({"data":[]})),
    ]));
    // The registered authenticator requires a credential; the request override
    // must work without one and must not leak to the next operation.
    let client = client("anthropic", transport.clone());
    let provider = client.provider::<AnthropicClient>("selected").unwrap();
    let local = RequestOptions {
        authenticator: Some(lingxi_llm_client::client::options::RequestAuthenticator(
            Arc::new(RequestAuth),
        )),
        account_scope: Some("account".into()),
        ..Default::default()
    };
    provider.files().list(None, &local).await.unwrap();
    provider
        .files()
        .upload(
            &UploadFile {
                filename: "note.txt".into(),
                media_type: "text/plain".into(),
                bytes: Bytes::from_static(b"hello"),
            },
            FilePurpose::ModelInput,
            &local,
        )
        .await
        .unwrap();
    provider
        .files()
        .list(None, &options("registry-key", "account"))
        .await
        .unwrap();
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    for request in &requests[..2] {
        assert!(request
            .headers
            .iter()
            .any(|(k, v)| k == "authorization" && v == "Bearer request-token"));
        assert!(!request
            .headers
            .iter()
            .any(|(k, _)| k == "x-injected-auth" || k == "x-api-key"));
    }
    assert!(requests[2]
        .headers
        .iter()
        .any(|(k, v)| k == "x-injected-auth" && v == "registry-key"));
    assert!(!requests[2]
        .headers
        .iter()
        .any(|(k, _)| k == "authorization"));
}
