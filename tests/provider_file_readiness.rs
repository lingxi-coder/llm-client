use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::files::{provider_file_endpoint_fingerprint, FileService, ProviderFileRef};
use lingxi_llm_client::protocol::{
    AttachmentRef, AuthStrategy, ChatRequest, ContentBlock, DocumentSource, LlmError,
    ProtocolFamily, ProviderFileSource, ProviderProfile, Region, Secret,
};
use lingxi_llm_client::{
    AttachmentResolver, Authenticator, CodecContext, EncodeRequest, GeminiCodec, HttpRequest,
    HttpResponse, LlmClientBuilder, OpenAiChatCodec, RequestMode, RequestOptions, StreamResponse,
    Transport, WireCodec,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn profile(provider: &str) -> ProviderProfile {
    let (base_url, protocol, model) = match provider {
        "google" => (
            "https://generativelanguage.googleapis.com/v1beta",
            "gemini_generate_content",
            "gemini-3.8-flash",
        ),
        "qwen" => (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "open_ai_chat",
            "qwen-long",
        ),
        _ => unreachable!(),
    };
    serde_json::from_value(json!({
        "profile_name": format!("{provider}-profile"),
        "provider_id": provider,
        "base_url": base_url,
        "protocol": protocol,
        "auth": "none",
        "regions": ["china_mainland", "international"],
        "models": [{
            "display_model": model,
            "request_model": model,
            "billing_model": model,
            "metadata": {"input_modalities": ["text", "file", "image", "video"]},
            "capability_support": {"vision": "supported", "documents": "supported"}
        }]
    }))
    .unwrap()
}

fn file(profile: &ProviderProfile, status: Option<&str>) -> ProviderFileRef {
    let gemini = profile.protocol == ProtocolFamily::GeminiGenerateContent;
    ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
        account_scope: Some("account-1".into()),
        protocol: profile.protocol,
        file_id: if gemini {
            "files/file-1".into()
        } else {
            "file-1".into()
        },
        uri: gemini
            .then(|| "https://generativelanguage.googleapis.com/v1beta/files/file-1".into())
            .or_else(|| Some("fileid://file-1".into())),
        filename: Some("report.pdf".into()),
        media_type: Some("application/pdf".into()),
        size_bytes: Some(3),
        expires_at: None,
        processing_status: status.map(str::to_owned),
        downloadable: None,
        purpose: (!gemini).then(|| "file-extract".into()),
    }
}

fn request(profile: &ProviderProfile, source: ProviderFileSource) -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model": profile.models[0].request_model,
        "messages": [{"role": "user", "content": []}]
    }))
    .unwrap();
    request.messages[0].content = vec![
        ContentBlock::Text {
            text: "Summarize this file.".into(),
            thought_signature: None,
            citations: None,
        },
        ContentBlock::Document {
            source: DocumentSource::ProviderFile { file: source },
            title: Some("report.pdf".into()),
        },
    ];
    request
}

fn codec(profile: &ProviderProfile) -> &'static dyn WireCodec {
    match profile.protocol {
        ProtocolFamily::GeminiGenerateContent => &GeminiCodec,
        ProtocolFamily::OpenAiChat => &OpenAiChatCodec,
        _ => unreachable!(),
    }
}

#[test]
fn known_gemini_and_qwen_readiness_states_are_checked_before_encoding() {
    for (provider, status, expected_processing_error) in [
        ("google", "PROCESSING", true),
        ("google", "FAILED", false),
        ("qwen", "uploaded", true),
        ("qwen", "processing", true),
        ("qwen", "error", false),
    ] {
        let profile = profile(provider);
        let request = request(&profile, file(&profile, Some(status)).model_reference());
        let context = CodecContext::new(
            &profile,
            &profile.models[0].request_model,
            RequestMode::Complete,
        )
        .with_file_scope(Some("account-1"));
        let error = codec(&profile)
            .encode_request(EncodeRequest::new(&request), &context)
            .unwrap_err();
        if expected_processing_error {
            let LlmError::ProviderFileProcessing { file, .. } = error else {
                panic!("{provider} {status} should remain resumable: {error}");
            };
            assert_eq!(file.processing_status.as_deref(), Some(status));
        } else {
            assert!(matches!(error, LlmError::InvalidRequest { .. }), "{error}");
        }
    }
}

#[test]
fn readiness_projection_round_trips_but_is_not_sent_and_unknown_states_stay_unknown() {
    for (provider, status) in [
        ("google", "ACTIVE"),
        ("qwen", "processed"),
        ("qwen", "future-provider-state"),
    ] {
        let profile = profile(provider);
        let source = file(&profile, Some(status)).model_reference();
        let encoded_source = serde_json::to_value(&source).unwrap();
        assert_eq!(encoded_source["processing_status"], status);
        let decoded: ProviderFileSource = serde_json::from_value(encoded_source).unwrap();
        assert_eq!(decoded, source);

        let request = request(&profile, source);
        let context = CodecContext::new(
            &profile,
            &profile.models[0].request_model,
            RequestMode::Complete,
        )
        .with_file_scope(Some("account-1"));
        let wire = codec(&profile)
            .encode_request(EncodeRequest::new(&request), &context)
            .unwrap();
        let body: Value = serde_json::from_slice(&wire.body).unwrap();
        assert!(!body.to_string().contains("processing_status"));
        assert!(!body.to_string().contains(status));
    }
}

#[derive(Default)]
struct QueueTransport {
    requests: Mutex<Vec<HttpRequest>>,
    responses: Mutex<VecDeque<HttpResponse>>,
}

impl QueueTransport {
    fn with_json(responses: Vec<Value>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            responses: Mutex::new(
                responses
                    .into_iter()
                    .map(|value| HttpResponse {
                        status: 200,
                        headers: vec![],
                        body: value.to_string().into(),
                    })
                    .collect(),
            ),
        }
    }
}

#[async_trait]
impl Transport for QueueTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let response =
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| LlmError::Transport {
                    message: "missing scripted response".into(),
                })?;
        Ok(response.into())
    }
}

#[tokio::test]
async fn gemini_resume_returns_the_authoritative_active_status() {
    let profile = profile("google");
    let http = QueueTransport::with_json(vec![json!({
        "name":"files/file-1",
        "uri":"https://generativelanguage.googleapis.com/v1beta/files/file-1",
        "mimeType":"application/pdf",
        "state":"ACTIVE"
    })]);
    let service = FileService::new(&http, &profile, None, None, Some("account-1"));

    let resumed = service
        .resume_gemini_processing(&file(&profile, Some("PROCESSING")).model_reference())
        .await
        .unwrap();
    assert_eq!(resumed.processing_status.as_deref(), Some("ACTIVE"));
    assert_eq!(
        resumed.model_reference().processing_status.as_deref(),
        Some("ACTIVE")
    );
    assert_eq!(http.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn get_preserves_an_omitted_status_and_explicit_null_clears_it() {
    let profile = profile("qwen");
    let http = QueueTransport::with_json(vec![
        json!({"id":"file-1","purpose":"file-extract"}),
        json!({"id":"file-1","purpose":"file-extract","status":null}),
    ]);
    let service = FileService::new(&http, &profile, None, None, Some("account-1"));
    let file = file(&profile, Some("processing"));

    let omitted = service.get(&file).await.unwrap();
    assert_eq!(omitted.status.as_deref(), Some("processing"));
    assert_eq!(
        omitted.file.processing_status.as_deref(),
        Some("processing")
    );

    let cleared = service.get(&omitted.file).await.unwrap();
    assert_eq!(cleared.status, None);
    assert_eq!(cleared.file.processing_status, None);
}

#[tokio::test]
async fn gemini_get_prefers_state_over_null_status_and_projects_active_updates() {
    let profile = profile("google");
    let http = QueueTransport::with_json(vec![
        json!({
            "name":"files/file-1",
            "uri":"https://generativelanguage.googleapis.com/v1beta/files/file-1",
            "status":null,
            "state":"PROCESSING"
        }),
        json!({
            "name":"files/file-1",
            "uri":"https://generativelanguage.googleapis.com/v1beta/files/file-1",
            "state":"ACTIVE"
        }),
    ]);
    let service = FileService::new(&http, &profile, None, None, Some("account-1"));
    let original = file(&profile, Some("PROCESSING"));

    let processing = service.get(&original).await.unwrap();
    assert_eq!(processing.status.as_deref(), Some("PROCESSING"));
    assert_eq!(
        processing.file.processing_status.as_deref(),
        Some("PROCESSING")
    );
    let active = service.get(&processing.file).await.unwrap();
    assert_eq!(active.status.as_deref(), Some("ACTIVE"));
    assert_eq!(active.file.processing_status.as_deref(), Some("ACTIVE"));
}

struct AttachmentBytes;

#[async_trait]
impl AttachmentResolver for AttachmentBytes {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        Ok(Bytes::from_static(b"pdf"))
    }
}

#[derive(Default)]
struct QwenAttachmentTransport {
    chat_body: Mutex<Option<Bytes>>,
}

fn response(value: Value) -> StreamResponse {
    HttpResponse {
        status: 200,
        headers: vec![],
        body: value.to_string().into(),
    }
    .into()
}

#[async_trait]
impl Transport for QwenAttachmentTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        match (request.method.as_str(), request.url.as_str()) {
            ("POST", url) if url.ends_with("/files") => Ok(response(json!({
                "id":"file-fe-auto", "filename":"report.pdf", "purpose":"file-extract",
                "bytes":3, "status":"uploaded"
            }))),
            ("GET", url) if url.ends_with("/files/file-fe-auto") => Ok(response(json!({
                "id":"file-fe-auto", "purpose":"file-extract", "status":"processed"
            }))),
            ("POST", url) if url.ends_with("/chat/completions") => {
                *self.chat_body.lock().unwrap() = Some(request.body.clone());
                Ok(response(json!({
                    "id":"chatcmpl-test", "model":"qwen-long",
                    "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
                })))
            }
            ("DELETE", url) if url.ends_with("/files/file-fe-auto") => Ok(response(json!({
                "id":"file-fe-auto", "deleted":true
            }))),
            _ => Err(LlmError::Transport {
                message: format!(
                    "unexpected Qwen request: {} {}",
                    request.method, request.url
                ),
            }),
        }
    }
}

#[tokio::test]
async fn automatic_qwen_attachment_uses_the_polled_processed_reference() {
    let profile = profile("qwen");
    let http = Arc::new(QwenAttachmentTransport::default());
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(AttachmentBytes));
    let client = builder.with_region(Region::ChinaMainland).build().unwrap();
    let request: ChatRequest = serde_json::from_value(json!({
        "model":"qwen-long",
        "messages":[{"role":"user","content":[
            {"type":"text","text":"Summarize this file."},
            {"type":"document","source":{"type":"attachment","attachment":{
                "attachment_id":"att-1","revision":"1","filename":"report.pdf",
                "media_type":"application/pdf","size_bytes":3
            }}}
        ]}]
    }))
    .unwrap();

    client
        .chat()
        .complete(&request, &RequestOptions::default())
        .await
        .unwrap();
    let body = http.chat_body.lock().unwrap().clone().unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("fileid://file-fe-auto"));
    assert!(!body.contains("uploaded"));
    assert!(!body.contains("processing_status"));
}

#[derive(Default)]
struct Counts {
    auth: AtomicUsize,
    transport: AtomicUsize,
}

struct CountingAuthenticator(Arc<Counts>);

#[async_trait]
impl Authenticator for CountingAuthenticator {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.0.auth.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct CountingTransport(Arc<Counts>);

#[async_trait]
impl Transport for CountingTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.transport.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::Transport {
            message: "unexpected request reached the transport".into(),
        })
    }
}

#[tokio::test]
async fn known_processing_status_fails_before_authentication_or_transport() {
    let mut profile = profile("google");
    profile.auth = AuthStrategy::ApiKey;
    let counts = Arc::new(Counts::default());
    let transport = Arc::new(CountingTransport(counts.clone()));
    let mut builder = LlmClientBuilder::with_transport(transport, std::slice::from_ref(&profile));
    builder.register_authenticator(
        AuthStrategy::ApiKey,
        Arc::new(CountingAuthenticator(counts.clone())),
    );
    let client = builder.with_region(Region::International).build().unwrap();
    let request = request(
        &profile,
        file(&profile, Some("PROCESSING")).model_reference(),
    );
    let options = RequestOptions {
        file_account_scope: Some("account-1".into()),
        ..Default::default()
    };

    assert!(matches!(
        client.chat().complete(&request, &options).await,
        Err(LlmError::ProviderFileProcessing { .. })
    ));
    assert_eq!(counts.auth.load(Ordering::SeqCst), 0);
    assert_eq!(counts.transport.load(Ordering::SeqCst), 0);
}
