use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::files::{
    FilePurpose, FileService, FileUploadError, ProviderFileRef, UploadFileStream,
};
use lingxi_llm_client::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{Authenticator, HttpRequest, HttpStreamRequest, StreamResponse, Transport};
use serde_json::json;
use std::{sync::Mutex, time::Duration};

const BUDGET: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
enum Behavior {
    PendingSend,
    PendingBody,
    SlowDownload,
    QuickList,
}

struct TestTransport {
    behavior: Behavior,
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for TestTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let content = request.url.ends_with("/content");
        self.requests.lock().unwrap().push(request);
        let body = match self.behavior {
            Behavior::PendingSend => return std::future::pending().await,
            Behavior::PendingBody => stream::pending().boxed(),
            Behavior::SlowDownload => {
                tokio::time::sleep(Duration::from_secs(6)).await;
                let body = if content {
                    Bytes::from_static(b"file contents")
                } else {
                    Bytes::from_static(br#"{"id":"file-1","downloadable":true}"#)
                };
                stream::iter([Ok(body)]).boxed()
            }
            Behavior::QuickList => {
                tokio::time::sleep(Duration::from_secs(6)).await;
                stream::iter([Ok(Bytes::from_static(br#"{"data":[]}"#))]).boxed()
            }
        };
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body,
        })
    }

    async fn send_stream(&self, _: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        std::future::pending().await
    }
}

struct PendingAuthenticator;

#[async_trait]
impl Authenticator for PendingAuthenticator {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        std::future::pending().await
    }
}

fn profile(auth: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"anthropic", "profile_name":"files", "auth":auth,
        "base_url":"https://api.anthropic.com", "protocol":"anthropic_messages"
    }))
    .unwrap()
}

fn transport(behavior: Behavior) -> TestTransport {
    TestTransport {
        behavior,
        requests: Mutex::new(Vec::new()),
    }
}

fn file(profile: &ProviderProfile) -> ProviderFileRef {
    ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        protocol: profile.protocol,
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            &profile.base_url,
        ),
        account_scope: Some("account".into()),
        file_id: "file-1".into(),
        uri: None,
        media_type: None,
        size_bytes: None,
        filename: None,
        purpose: None,
        processing_status: None,
        downloadable: None,
        expires_at: None,
    }
}

#[tokio::test(start_paused = true)]
async fn total_timeout_includes_authentication_before_dispatch() {
    let profile = profile("api_key");
    let transport = transport(Behavior::PendingSend);
    let service = FileService::new(
        &transport,
        &profile,
        Some(&PendingAuthenticator),
        None,
        Some("account"),
    )
    .with_timeout(BUDGET)
    .unwrap();
    let start = tokio::time::Instant::now();
    let result = service.list(None).await;
    assert!(matches!(result, Err(LlmError::TransportTimeout { .. })));
    assert_eq!(start.elapsed(), BUDGET);
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test(start_paused = true)]
async fn total_timeout_bounds_custom_transport_and_response_body() {
    for behavior in [Behavior::PendingSend, Behavior::PendingBody] {
        let profile = profile("none");
        let transport = transport(behavior);
        let service = FileService::new(&transport, &profile, None, None, Some("account"))
            .with_timeout(BUDGET)
            .unwrap();
        let start = tokio::time::Instant::now();
        let result = service.list(None).await;
        assert!(matches!(result, Err(LlmError::TransportTimeout { .. })));
        assert_eq!(start.elapsed(), BUDGET);
        assert_eq!(transport.requests.lock().unwrap()[0].timeout, Some(BUDGET));
    }
}

#[tokio::test(start_paused = true)]
async fn download_metadata_and_content_share_one_total_budget() {
    let profile = profile("none");
    let transport = transport(Behavior::SlowDownload);
    let service = FileService::new(&transport, &profile, None, None, Some("account"))
        .with_timeout(BUDGET)
        .unwrap();
    let start = tokio::time::Instant::now();
    let result = service.download(&file(&profile)).await;
    assert!(
        matches!(result, Err(LlmError::TransportTimeout { .. })),
        "{result:?}"
    );
    assert_eq!(start.elapsed(), BUDGET);
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].timeout, Some(Duration::from_secs(4)));
}

#[tokio::test(start_paused = true)]
async fn separate_operations_receive_fresh_budgets() {
    let profile = profile("none");
    let transport = transport(Behavior::QuickList);
    let service = FileService::new(&transport, &profile, None, None, Some("account"))
        .with_timeout(BUDGET)
        .unwrap();
    service.list(None).await.unwrap();
    tokio::time::sleep(Duration::from_secs(30)).await;
    service.list(None).await.unwrap();
    assert!(transport
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|request| request.timeout == Some(BUDGET)));
}

#[tokio::test(start_paused = true)]
async fn explicit_deadline_wins_over_fresh_operation_timeout() {
    let profile = profile("none");
    let transport = transport(Behavior::PendingSend);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    let service = FileService::new(&transport, &profile, None, None, Some("account"))
        .with_timeout(BUDGET)
        .unwrap()
        .with_deadline(deadline.into_std());
    let start = tokio::time::Instant::now();
    assert!(matches!(
        service.list(None).await,
        Err(LlmError::TransportTimeout { .. })
    ));
    assert_eq!(start.elapsed(), Duration::from_secs(3));
}

#[tokio::test(start_paused = true)]
async fn upload_timeout_preserves_unknown_outcome_after_dispatch() {
    let profile = profile("none");
    let transport = transport(Behavior::PendingSend);
    let service = FileService::new(&transport, &profile, None, None, Some("account"))
        .with_timeout(BUDGET)
        .unwrap();
    let result = service
        .upload_stream(
            UploadFileStream::from_bytes(
                "document.pdf",
                "application/pdf",
                Bytes::from_static(b"pdf"),
            ),
            FilePurpose::ModelInput,
        )
        .await;
    assert!(
        matches!(result, Err(FileUploadError::OutcomeUnknown { source, .. }) if matches!(*source, LlmError::TransportTimeout { .. }))
    );
}

#[test]
fn zero_operation_timeout_is_rejected() {
    let profile = profile("none");
    let transport = transport(Behavior::PendingSend);
    assert!(matches!(
        FileService::new(&transport, &profile, None, None, Some("account"))
            .with_timeout(Duration::ZERO),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[tokio::test(start_paused = true)]
async fn gemini_processing_timeout_keeps_the_pending_file_reference() {
    struct GeminiTransport;
    #[async_trait]
    impl Transport for GeminiTransport {
        async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
            if request.method == "GET" {
                return std::future::pending().await;
            }
            let (headers, body) = if request.url.ends_with("?upload_id=one") {
                (vec![], Bytes::from_static(br#"{"file":{"name":"files/pending","uri":"https://generativelanguage.googleapis.com/v1beta/files/pending","mimeType":"application/pdf","state":"PROCESSING"}}"#))
            } else {
                (vec![("x-goog-upload-url".into(), "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=one".into())], Bytes::new())
            };
            Ok(StreamResponse {
                status: 200,
                headers,
                body: stream::iter([Ok(body)]).boxed(),
            })
        }
    }
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"google", "profile_name":"gemini", "auth":"none",
        "base_url":"https://generativelanguage.googleapis.com/v1beta", "protocol":"gemini_generate_content"
    })).unwrap();
    let service = FileService::new(&GeminiTransport, &profile, None, None, Some("account"))
        .with_timeout(BUDGET)
        .unwrap();
    let start = tokio::time::Instant::now();
    let result = service
        .upload(
            &lingxi_llm_client::files::UploadFile {
                filename: "document.pdf".into(),
                media_type: "application/pdf".into(),
                bytes: Bytes::from_static(b"pdf"),
            },
            FilePurpose::ModelInput,
        )
        .await;
    assert!(
        matches!(result, Err(LlmError::ProviderFileProcessing { .. })),
        "{result:?}"
    );
    assert!(start.elapsed() <= BUDGET);
    assert!(start.elapsed() >= Duration::from_secs(9));
}

#[tokio::test(start_paused = true)]
async fn provider_files_pass_request_timeout_to_the_shared_file_core() {
    let profile = profile("none");
    let file = file(&profile);
    let transport = std::sync::Arc::new(transport(Behavior::SlowDownload));
    let client = lingxi_llm_client::LlmClientBuilder::with_transport(transport.clone(), &[profile])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let provider = client
        .provider::<lingxi_llm_client::providers::AnthropicClient>("files")
        .unwrap();
    let options = lingxi_llm_client::RequestOptions {
        file_account_scope: Some("account".into()),
        total_timeout: Some(BUDGET),
        ..Default::default()
    };
    let start = tokio::time::Instant::now();
    let result = provider.files().download(&file, &options).await;
    assert!(
        matches!(result, Err(LlmError::TransportTimeout { .. })),
        "{result:?}"
    );
    assert_eq!(start.elapsed(), BUDGET);
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].timeout, Some(Duration::from_secs(4)));
}
