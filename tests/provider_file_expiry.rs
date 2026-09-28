use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::files::{provider_file_endpoint_fingerprint, FileService, ProviderFileRef};
use lingxi_llm_client::protocol::{
    AttachmentRef, AuthStrategy, ChatRequest, ContentBlock, ConversationMessage, DocumentSource,
    ImageSource, LlmError, MessageRole, ProtocolFamily, ProviderFileSource, ProviderProfile,
    Region, Secret, VideoSource,
};
use lingxi_llm_client::providers::anthropic::types::AnthropicCodeExecutionConfig;
use lingxi_llm_client::transport::Clock;
use lingxi_llm_client::{
    AnthropicMessagesCodec, Authenticator, CodecContext, EncodeRequest, GeminiCodec, HttpRequest,
    HttpResponse, LlmClientBuilder, OpenAiChatCodec, OpenAiResponsesCodec, RequestMode,
    RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const NOW: u64 = 2_000_000_000;
const SCOPE: &str = "account-a";

fn profile(protocol: ProtocolFamily) -> ProviderProfile {
    let (provider, endpoint, model) = match protocol {
        ProtocolFamily::AnthropicMessages => {
            ("anthropic", "https://api.anthropic.com", "claude-opus-5-5")
        }
        ProtocolFamily::GeminiGenerateContent => (
            "google",
            "https://generativelanguage.googleapis.com/v1beta",
            "gemini-3.8-flash",
        ),
        _ => ("openai", "https://api.openai.com/v1", "gpt-4.1"),
    };
    serde_json::from_value(json!({
        "provider_id":provider,"profile_name":provider,"base_url":endpoint,"protocol":protocol,
        "auth":"none","regions":["international"],"models":[{
            "display_model":model,"request_model":model,"billing_model":model,
            "metadata":{"input_modalities":["text","file","image","video"]},
            "capability_support":{"vision":"supported","documents":"supported"}
        }]
    }))
    .unwrap()
}

fn file(profile: &ProviderProfile, expires_at: Option<&str>) -> ProviderFileRef {
    ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
        account_scope: Some(SCOPE.into()),
        protocol: profile.protocol,
        file_id: "file_123".into(),
        uri: (profile.protocol == ProtocolFamily::GeminiGenerateContent)
            .then(|| "https://generativelanguage.googleapis.com/v1beta/files/file_123".into()),
        filename: Some("doc.pdf".into()),
        media_type: Some("application/pdf".into()),
        size_bytes: Some(8),
        expires_at: expires_at.map(str::to_owned),
        processing_status: None,
        downloadable: Some(false),
        purpose: (profile.provider_id.as_str() == "openai").then(|| "user_data".into()),
    }
}

fn request(profile: &ProviderProfile, source: ProviderFileSource) -> ChatRequest {
    let mut request: ChatRequest =
        serde_json::from_value(json!({"model":profile.models[0].request_model,"messages":[]}))
            .unwrap();
    request.messages = vec![ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::Document {
            source: DocumentSource::ProviderFile { file: source },
            title: None,
        }],
    }];
    request
}

fn context(profile: &ProviderProfile) -> CodecContext {
    CodecContext::new(
        profile,
        &profile.models[0].request_model,
        RequestMode::Complete,
    )
    .with_file_scope(Some(SCOPE))
    .with_file_validation_time(UNIX_EPOCH + Duration::from_secs(NOW))
}

fn codec(protocol: ProtocolFamily) -> &'static dyn WireCodec {
    match protocol {
        ProtocolFamily::OpenAiChat => &OpenAiChatCodec,
        ProtocolFamily::OpenAiResponses => &OpenAiResponsesCodec,
        ProtocolFamily::AnthropicMessages => &AnthropicMessagesCodec,
        ProtocolFamily::GeminiGenerateContent => &GeminiCodec,
        _ => unreachable!(),
    }
}

const PROTOCOLS: [ProtocolFamily; 4] = [
    ProtocolFamily::OpenAiChat,
    ProtocolFamily::OpenAiResponses,
    ProtocolFamily::AnthropicMessages,
    ProtocolFamily::GeminiGenerateContent,
];

#[test]
fn expiry_survives_model_projection_and_serialization_without_rewriting_timestamp() {
    let profile = profile(ProtocolFamily::AnthropicMessages);
    for raw in [
        "2033-05-18T03:33:20.125+00:00",
        "2000000001",
        "invalid-but-preserved",
    ] {
        let source = file(&profile, Some(raw)).model_reference();
        assert_eq!(source.expires_at.as_deref(), Some(raw));
        let encoded = serde_json::to_value(&source).unwrap();
        assert_eq!(encoded["expires_at"], raw);
        let decoded: ProviderFileSource = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, source);
    }
    let source = file(&profile, None).model_reference();
    assert!(serde_json::to_value(source)
        .unwrap()
        .get("expires_at")
        .is_none());
}

#[test]
fn every_file_wire_rejects_expired_equal_or_malformed_timestamps() {
    for protocol in PROTOCOLS {
        let profile = profile(protocol);
        for expiry in [
            "1999999999",
            "2000000000",
            "1970-01-01T00:00:00Z",
            "",
            "not-a-date",
            "true",
            "{}",
            "999999999999999999999999999999",
        ] {
            let request = request(&profile, file(&profile, Some(expiry)).model_reference());
            let error = codec(protocol)
                .encode_request(EncodeRequest::new(&request), &context(&profile))
                .unwrap_err();
            assert!(
                matches!(error, LlmError::InvalidRequest { .. }),
                "{protocol:?}, {expiry:?}: {error}"
            );
        }
    }
}

#[test]
fn future_or_absent_expiry_is_local_metadata_and_never_sent_on_wire() {
    for protocol in PROTOCOLS {
        let profile = profile(protocol);
        for expiry in [
            None,
            Some("2000000001"),
            Some("2099-01-01T01:00:00.123+01:00"),
        ] {
            let request = request(&profile, file(&profile, expiry).model_reference());
            let wire = codec(protocol)
                .encode_request(EncodeRequest::new(&request), &context(&profile))
                .unwrap();
            let body: Value = serde_json::from_slice(&wire.body).unwrap();
            assert!(!body.to_string().contains("expires_at"));
            assert!(!body.to_string().contains(SCOPE));
        }
    }
}

#[test]
fn image_video_and_sandbox_uploads_share_expiry_validation() {
    let profile = profile(ProtocolFamily::GeminiGenerateContent);
    let mut image = file(&profile, Some("2000000000")).model_reference();
    image.media_type = Some("image/png".into());
    let mut video = image.clone();
    video.media_type = Some("video/mp4".into());
    for block in [
        ContentBlock::Image {
            source: ImageSource::ProviderFile { file: image },
        },
        ContentBlock::Video {
            source: VideoSource::ProviderFile { file: video },
        },
    ] {
        let mut request = request(&profile, file(&profile, None).model_reference());
        request.messages[0].content = vec![block];
        assert!(matches!(
            GeminiCodec.encode_request(EncodeRequest::new(&request), &context(&profile)),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
    let anthropic = crate::profile(ProtocolFamily::AnthropicMessages);
    let mut request = request(&anthropic, file(&anthropic, None).model_reference());
    request.messages = vec![ConversationMessage::user_text("Analyze the CSV")];
    let mut source = file(&anthropic, Some("2000000000")).model_reference();
    source.media_type = Some("text/csv".into());
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
            AnthropicCodeExecutionConfig {
                files: vec![source],
                skills: vec![],
                ..Default::default()
            },
        )
        .into(),
    );
    assert!(matches!(
        AnthropicMessagesCodec.encode_request(EncodeRequest::new(&request), &context(&anthropic)),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[derive(Default)]
struct Http {
    requests: Mutex<Vec<HttpRequest>>,
    responses: Mutex<VecDeque<HttpResponse>>,
}
impl Http {
    fn new(responses: Vec<Value>) -> Self {
        Self {
            requests: Mutex::new(vec![]),
            responses: Mutex::new(
                responses
                    .into_iter()
                    .map(|body| HttpResponse {
                        status: 200,
                        headers: vec![],
                        body: body.to_string().into(),
                    })
                    .collect(),
            ),
        }
    }
}
#[async_trait]
impl Transport for Http {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let response = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        Ok(StreamResponse {
            status: response.status,
            headers: response.headers,
            body: stream::iter(vec![Ok(response.body)]).boxed(),
        })
    }
}

#[tokio::test]
async fn malformed_metadata_is_preserved_as_invalid_instead_of_becoming_missing_expiry() {
    let profile = profile(ProtocolFamily::AnthropicMessages);
    for raw in [
        json!(true),
        json!({"bad":"shape"}),
        json!(""),
        json!("not-a-date"),
    ] {
        let transport = Http::new(vec![
            json!({"id":"file_123","mime_type":"application/pdf","expires_at":raw}),
        ]);
        let service = FileService::new(&transport, &profile, None, None, Some(SCOPE));
        let metadata = service.get(&file(&profile, None)).await.unwrap();
        assert!(metadata.file.expires_at.is_some());
        let request = request(&profile, metadata.file.model_reference());
        assert!(matches!(
            AnthropicMessagesCodec.encode_request(EncodeRequest::new(&request), &context(&profile)),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}

#[tokio::test]
async fn refreshing_metadata_keeps_omitted_expiry_but_allows_explicit_null_to_clear() {
    let profile = profile(ProtocolFamily::AnthropicMessages);
    let transport = Http::new(vec![
        json!({"id":"file_123","mime_type":"application/pdf"}),
        json!({"id":"file_123","mime_type":"application/pdf","expires_at":null}),
    ]);
    let service = FileService::new(&transport, &profile, None, None, Some(SCOPE));
    let original = file(&profile, Some("2000000000"));
    let kept = service.get(&original).await.unwrap();
    assert_eq!(kept.file.expires_at, original.expires_at);
    let cleared = service.get(&original).await.unwrap();
    assert_eq!(cleared.file.expires_at, None);
}

struct TestClock(AtomicU64);
impl Clock for TestClock {
    fn now(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(self.0.load(Ordering::SeqCst))
    }
}
#[derive(Default)]
struct Counts {
    resolve: AtomicUsize,
    auth: AtomicUsize,
}
struct Resolver {
    counts: Arc<Counts>,
    clock: Arc<TestClock>,
    advance: bool,
}
#[async_trait]
impl lingxi_llm_client::AttachmentResolver for Resolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.counts.resolve.fetch_add(1, Ordering::SeqCst);
        if self.advance {
            self.clock.0.store(NOW + 1, Ordering::SeqCst);
        }
        Ok(Bytes::from_static(b"document"))
    }
}
struct Auth {
    counts: Arc<Counts>,
    clock: Arc<TestClock>,
    advance: bool,
}
#[async_trait]
impl Authenticator for Auth {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.counts.auth.fetch_add(1, Ordering::SeqCst);
        if self.advance && !request.url.ends_with("/files") {
            self.clock.0.store(NOW + 1, Ordering::SeqCst);
        }
        Ok(())
    }
}
fn attachment() -> ContentBlock {
    ContentBlock::Document {
        source: DocumentSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "doc".into(),
                revision: "1".into(),
                filename: "doc.pdf".into(),
                media_type: "application/pdf".into(),
                size_bytes: 8,
            },
        },
        title: None,
    }
}

#[tokio::test]
async fn expired_references_fail_before_resolver_auth_or_http_for_every_route() {
    for protocol in PROTOCOLS {
        let mut profile = profile(protocol);
        profile.auth = AuthStrategy::ApiKey;
        let transport = Arc::new(Http::default());
        let counts = Arc::new(Counts::default());
        let clock = Arc::new(TestClock(AtomicU64::new(NOW)));
        let mut builder =
            LlmClientBuilder::with_transport(transport.clone(), std::slice::from_ref(&profile));
        builder.with_clock(clock.clone());
        builder.with_attachment_resolver(Arc::new(Resolver {
            counts: counts.clone(),
            clock: clock.clone(),
            advance: false,
        }));
        builder.register_authenticator(
            AuthStrategy::ApiKey,
            Arc::new(Auth {
                counts: counts.clone(),
                clock,
                advance: false,
            }),
        );
        let client = builder.with_region(Region::International).build().unwrap();
        let mut request = request(
            &profile,
            file(&profile, Some("2000000000")).model_reference(),
        );
        request.messages[0].content.push(attachment());
        let options = RequestOptions {
            file_account_scope: Some(SCOPE.into()),
            ..Default::default()
        };
        assert!(matches!(
            client.chat().complete(&request, &options).await,
            Err(LlmError::InvalidRequest { .. })
        ));
        assert!(matches!(
            client.chat().stream(&request, &options).await,
            Err(LlmError::InvalidRequest { .. })
        ));
        assert_eq!(counts.resolve.load(Ordering::SeqCst), 0);
        assert_eq!(counts.auth.load(Ordering::SeqCst), 0);
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn expiry_is_rechecked_after_resolver_and_after_authentication_awaits() {
    for during_resolver in [true, false] {
        let mut profile = profile(ProtocolFamily::OpenAiResponses);
        profile.auth = AuthStrategy::ApiKey;
        let transport = Arc::new(Http::default());
        let counts = Arc::new(Counts::default());
        let clock = Arc::new(TestClock(AtomicU64::new(NOW)));
        let mut builder =
            LlmClientBuilder::with_transport(transport.clone(), std::slice::from_ref(&profile));
        builder.with_clock(clock.clone());
        builder.with_attachment_resolver(Arc::new(Resolver {
            counts: counts.clone(),
            clock: clock.clone(),
            advance: during_resolver,
        }));
        builder.register_authenticator(
            AuthStrategy::ApiKey,
            Arc::new(Auth {
                counts: counts.clone(),
                clock,
                advance: !during_resolver,
            }),
        );
        let client = builder.with_region(Region::International).build().unwrap();
        let mut request = request(
            &profile,
            file(&profile, Some("2000000001")).model_reference(),
        );
        if during_resolver {
            request.messages[0].content.push(attachment());
        }
        let options = RequestOptions {
            file_account_scope: Some(SCOPE.into()),
            ..Default::default()
        };
        assert!(matches!(
            client.chat().complete(&request, &options).await,
            Err(LlmError::InvalidRequest { .. })
        ));
        assert_eq!(
            counts.resolve.load(Ordering::SeqCst),
            usize::from(during_resolver)
        );
        assert_eq!(
            counts.auth.load(Ordering::SeqCst),
            usize::from(!during_resolver)
        );
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn automatic_upload_is_replaced_when_provider_expiry_reaches_the_clock() {
    let mut profile = profile(ProtocolFamily::OpenAiResponses);
    profile.auth = AuthStrategy::ApiKey;
    let transport = Arc::new(Http::new(vec![
        json!({"id":"file_auto","bytes":8,"filename":"doc.pdf","purpose":"user_data","expires_at":NOW+1}),
        json!({"id":"resp_1","model":"gpt-4.1","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":1}}),
        json!({"id":"file_replacement","bytes":8,"filename":"doc.pdf","purpose":"user_data","expires_at":NOW+100}),
        json!({"id":"resp_2","model":"gpt-4.1","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":1}}),
    ]));
    let counts = Arc::new(Counts::default());
    let clock = Arc::new(TestClock(AtomicU64::new(NOW)));
    let mut builder =
        LlmClientBuilder::with_transport(transport.clone(), std::slice::from_ref(&profile));
    builder.with_clock(clock.clone());
    builder.with_attachment_resolver(Arc::new(Resolver {
        counts: counts.clone(),
        clock: clock.clone(),
        advance: false,
    }));
    builder.register_authenticator(
        AuthStrategy::ApiKey,
        Arc::new(Auth {
            counts: counts.clone(),
            clock: clock.clone(),
            advance: false,
        }),
    );
    let client = builder.with_region(Region::International).build().unwrap();
    let mut request = request(&profile, file(&profile, None).model_reference());
    request.messages[0].content = vec![attachment()];
    let options = RequestOptions {
        file_account_scope: Some(SCOPE.into()),
        ..Default::default()
    };
    client.chat().complete(&request, &options).await.unwrap();
    clock.0.store(NOW + 1, Ordering::SeqCst);
    client.chat().complete(&request, &options).await.unwrap();
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert!(requests[0].url.ends_with("/files"));
    assert!(requests[1].url.ends_with("/responses"));
    assert!(requests[2].url.ends_with("/files"));
    assert!(requests[3].url.ends_with("/responses"));
    let body: Value = serde_json::from_slice(&requests[3].body).unwrap();
    assert_eq!(
        body["input"][0]["content"][0]["file_id"],
        "file_replacement"
    );
    assert_eq!(counts.auth.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn automatic_file_that_expires_during_model_auth_never_reaches_model_transport() {
    let mut profile = profile(ProtocolFamily::OpenAiResponses);
    profile.auth = AuthStrategy::ApiKey;
    let transport = Arc::new(Http::new(vec![
        json!({"id":"file_auto","bytes":8,"filename":"doc.pdf","purpose":"user_data","expires_at":NOW+1}),
    ]));
    let counts = Arc::new(Counts::default());
    let clock = Arc::new(TestClock(AtomicU64::new(NOW)));
    let mut builder =
        LlmClientBuilder::with_transport(transport.clone(), std::slice::from_ref(&profile));
    builder.with_clock(clock.clone());
    builder.with_attachment_resolver(Arc::new(Resolver {
        counts: counts.clone(),
        clock: clock.clone(),
        advance: false,
    }));
    builder.register_authenticator(
        AuthStrategy::ApiKey,
        Arc::new(Auth {
            counts: counts.clone(),
            clock,
            advance: true,
        }),
    );
    let client = builder.with_region(Region::International).build().unwrap();
    let mut request = request(&profile, file(&profile, None).model_reference());
    request.messages[0].content = vec![attachment()];
    let options = RequestOptions {
        file_account_scope: Some(SCOPE.into()),
        ..Default::default()
    };
    assert!(matches!(
        client.chat().complete(&request, &options).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].url.ends_with("/files"));
}
