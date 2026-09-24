//! Gate 32: a request walks `ResolvedRoute::connection_chain`, and gate 30's
//! half that this crate can hold on its own — nothing here knows a provider by
//! name, so two profiles that differ only in settings behave differently.
//!
//! The fakes are a `Transport` and a `WireCodec`, per the review: a fake
//! provider is built at the transport/codec layer, never by stubbing the
//! client itself.
use lingxi_llm_client::protocol::PricingContext;

#[path = "support/wire_api.rs"]
mod wire_api;

use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on;
use futures::stream;
use lingxi_llm_client::protocol::{
    AttachmentRef, CompletionRequest, CompletionResponse, ContentBlock, ConversationMessage,
    DocumentSource, ImageSource, LlmError, MessageRole, ProtocolFamily, ProviderProfile,
    ResponseId, StopReason, StreamEvent, ToolChoice, Usage, WebSearchConfig,
};
use lingxi_llm_client::{
    builtin_providers, HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions, ResolveError,
    StreamDecoder, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// --- fakes -----------------------------------------------------------------

/// Answers per URL: the script says what each base URL does. Records the order
/// it was called in, which is the thing gate 32 is actually about.
#[derive(Default)]
struct ScriptedTransport {
    answers: Mutex<Vec<(String, Result<u16, LlmError>)>>,
    seen: Mutex<Vec<String>>,
    calls: AtomicUsize,
    seen_auth: Mutex<Vec<Option<String>>>,
    seen_bodies: Mutex<Vec<Vec<u8>>>,
    seen_timeouts: Mutex<Vec<Option<std::time::Duration>>>,
}

impl ScriptedTransport {
    fn new(answers: Vec<(&str, Result<u16, LlmError>)>) -> Arc<Self> {
        Arc::new(Self {
            answers: Mutex::new(
                answers
                    .into_iter()
                    .map(|(u, r)| (u.to_owned(), r))
                    .collect(),
            ),
            ..Default::default()
        })
    }

    fn from_owned(answers: Vec<(String, Result<u16, LlmError>)>) -> Arc<Self> {
        Arc::new(Self {
            answers: Mutex::new(answers),
            ..Default::default()
        })
    }

    fn answer_for(&self, url: &str) -> Result<HttpResponse, LlmError> {
        self.seen.lock().unwrap().push(url.to_owned());
        self.calls.fetch_add(1, Ordering::SeqCst);
        let answers = self.answers.lock().unwrap();
        let hit = answers
            .iter()
            .find(|(u, _)| url.starts_with(u.as_str()))
            .map(|(_, r)| r.clone());
        match hit {
            Some(Ok(status)) => Ok(HttpResponse {
                status,
                // Real providers report quota state in headers on the success
                // path too, so the fake does the same: it is the only way to
                // show they survive to the caller of a *stream*.
                headers: vec![
                    (
                        "Anthropic-RateLimit-Unified-5h-Utilization".to_owned(),
                        "0.42".to_owned(),
                    ),
                    ("retry-after".to_owned(), "7".to_owned()),
                ],
                body: Bytes::from_static(b"{\"text\":\"hi\"}"),
            }),
            Some(Err(e)) => Err(e),
            None => Err(LlmError::Transport {
                message: format!("no script for {url}"),
            }),
        }
    }

    fn hops(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }

    fn auth_headers(&self) -> Vec<Option<String>> {
        self.seen_auth.lock().unwrap().clone()
    }

    fn bodies(&self) -> Vec<Vec<u8>> {
        self.seen_bodies.lock().unwrap().clone()
    }

    fn timeouts(&self) -> Vec<Option<std::time::Duration>> {
        self.seen_timeouts.lock().unwrap().clone()
    }

    fn record_request(&self, req: &HttpRequest) {
        self.seen_auth.lock().unwrap().push(
            req.headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.clone()),
        );
        self.seen_bodies.lock().unwrap().push(req.body.to_vec());
        self.seen_timeouts.lock().unwrap().push(req.timeout);
    }
}

#[async_trait]
impl Transport for ScriptedTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        if serde_json::from_slice::<serde_json::Value>(&request.body)
            .ok()
            .and_then(|body| body.get("stream").and_then(serde_json::Value::as_bool))
            == Some(true)
        {
            self.stream_response(request).await
        } else {
            self.response(request).await.map(Into::into)
        }
    }
}
impl ScriptedTransport {
    async fn response(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.record_request(&req);
        self.answer_for(&req.url)
    }
    async fn stream_response(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.record_request(&req);
        let answered = self.answer_for(&req.url)?;
        let frames: Vec<Result<Bytes, LlmError>> = vec![
            Ok(Bytes::from_static(b"one")),
            Ok(Bytes::from_static(b"two")),
        ];
        Ok(StreamResponse {
            status: answered.status,
            headers: answered.headers,
            body: Box::pin(stream::iter(frames)),
        })
    }
}

struct InterruptedErrorBodyTransport {
    seen: Mutex<Vec<String>>,
    backup_succeeds: bool,
}

impl InterruptedErrorBodyTransport {
    fn new(backup_succeeds: bool) -> Arc<Self> {
        Arc::new(Self {
            seen: Mutex::new(Vec::new()),
            backup_succeeds,
        })
    }

    fn hops(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for InterruptedErrorBodyTransport {
    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.seen.lock().unwrap().push(req.url.clone());
        let backup = req.url.starts_with("https://two.test");
        let succeeds = backup && self.backup_succeeds;
        let body: Vec<Result<Bytes, LlmError>> = if succeeds {
            Vec::new()
        } else {
            vec![
                Ok(Bytes::from_static(
                    br#"{"error":{"message":"quota exceeded"}}"#,
                )),
                Err(LlmError::StreamInterrupted {
                    message: "error body connection reset".to_owned(),
                }),
            ]
        };
        Ok(StreamResponse {
            status: if succeeds { 200 } else { 429 },
            headers: vec![("Retry-After".to_owned(), "7".to_owned())],
            body: Box::pin(stream::iter(body)),
        })
    }
}

type RecordedFileRequest = (String, String, Option<String>, Vec<u8>);

struct FileFailoverTransport {
    fail_first_completion: bool,
    missing_first_file: bool,
    uploads: AtomicUsize,
    completions: AtomicUsize,
    requests: Mutex<Vec<RecordedFileRequest>>,
}

impl FileFailoverTransport {
    fn new(fail_first_completion: bool) -> Arc<Self> {
        Arc::new(Self {
            fail_first_completion,
            missing_first_file: false,
            uploads: AtomicUsize::new(0),
            completions: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn missing_first_file() -> Arc<Self> {
        Arc::new(Self {
            fail_first_completion: false,
            missing_first_file: true,
            uploads: AtomicUsize::new(0),
            completions: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl Transport for FileFailoverTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl FileFailoverTransport {
    async fn response(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        let authorization = req
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.clone());
        self.requests.lock().unwrap().push((
            req.method.clone(),
            req.url.clone(),
            authorization,
            req.body.to_vec(),
        ));
        if req.url.ends_with("/files") {
            let id = self.uploads.fetch_add(1, Ordering::SeqCst) + 1;
            return Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".into(), "application/json".into())],
                body: Bytes::from(format!(r#"{{"id":"file-{id}"}}"#)),
            });
        }
        if req.url.ends_with("/responses") {
            let attempt = self.completions.fetch_add(1, Ordering::SeqCst);
            return if self.fail_first_completion && attempt == 0 {
                Ok(HttpResponse {
                    status: 429,
                    headers: vec![("content-type".into(), "application/json".into())],
                    body: Bytes::from_static(
                        br#"{"error":{"type":"rate_limit_error","message":"try backup"}}"#,
                    ),
                })
            } else if self.missing_first_file && attempt == 0 {
                Ok(HttpResponse {
                    status: 404,
                    headers: vec![("content-type".into(), "application/json".into())],
                    body: Bytes::from_static(
                        br#"{"error":{"message":"File file-1 not found","type":"invalid_request_error"}}"#,
                    ),
                })
            } else {
                Ok(HttpResponse {
                    status: 200,
                    headers: vec![("content-type".into(), "application/json".into())],
                    body: Bytes::from_static(
                        br#"{"status":"completed","model":"gpt-test","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}"#,
                    ),
                })
            };
        }
        Err(LlmError::Transport {
            message: format!("unexpected test URL: {}", req.url),
        })
    }
}

/// Encodes to `<base_url>/chat`, decodes 200 to one text block. One frame
/// decodes to one `TextDelta`, so a test can count frames as events.
struct FakeCodec;

#[async_trait]
impl WireCodec for FakeCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
    }

    fn encode_request(
        &self,
        input: lingxi_llm_client::EncodeRequest<'_>,
        context: &lingxi_llm_client::CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let req = input.request();
        let profile = context.profile();
        let route = &wire_api::route(context);
        let opts = &wire_api::options(context);

        // The endpoint comes from the connection's own profile, which is why
        // the codec is handed one: it is registered per protocol family and
        // shared by every profile using it.
        let base = &profile.base_url;
        Ok(HttpRequest {
            method: "POST".to_owned(),
            url: format!("{base}/chat"),
            headers: vec![],
            body: Bytes::from(
                serde_json::to_vec(
                    &json!({ "model": route.request_model, "n": req.messages.len(), "web_search": req.web_search, "stream": opts.stream }),
                )
                .unwrap(),
            ),
            timeout: None,
        })
    }
    fn decode_response(
        &self,
        resp: &HttpResponse,
        _context: &lingxi_llm_client::CodecContext,
    ) -> Result<CompletionResponse, LlmError> {
        if resp.status != 200 {
            return Err(LlmError::ProviderInternal {
                message: format!("status {}", resp.status),
            });
        }
        Ok(CompletionResponse {
            inference: Default::default(),
            web_search: None,
            file_search: None,
            message: ConversationMessage {
                role: MessageRole::Assistant,
                content: vec![ContentBlock::Text {
                    text: "hi".to_owned(),
                    thought_signature: None,
                }],
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default().into(),
            model: "fake".to_owned(),
            response_id: None,
            executed_profile: None,
        })
    }
    fn stream_decoder(&self, _context: &lingxi_llm_client::CodecContext) -> Box<dyn StreamDecoder> {
        Box::new(FakeDecoder { blocks: 0 })
    }
}

/// The same fake on the wire that keeps turn state at the endpoint. Only the
/// family differs — so a test using it isolates what the *client* does with a
/// continuation, not what a real encoder does.
struct FakeStatefulCodec;

#[async_trait]
impl WireCodec for FakeStatefulCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiResponses
    }

    fn encode_request(
        &self,
        input: lingxi_llm_client::EncodeRequest<'_>,
        context: &lingxi_llm_client::CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let req = input.request();
        let profile = context.profile();
        let route = &wire_api::route(context);
        let opts = &wire_api::options(context);

        FakeCodec.encode_request(
            lingxi_llm_client::EncodeRequest::new(req),
            &wire_api::context(profile, &(route).request_model, opts),
        )
    }
    fn decode_response(
        &self,
        resp: &HttpResponse,
        _context: &lingxi_llm_client::CodecContext,
    ) -> Result<CompletionResponse, LlmError> {
        FakeCodec.decode_response(resp, &wire_api::decode_context())
    }
    fn stream_decoder(&self, _context: &lingxi_llm_client::CodecContext) -> Box<dyn StreamDecoder> {
        FakeCodec.stream_decoder(&wire_api::decode_context())
    }
}

struct FakeDecoder {
    blocks: usize,
}

impl StreamDecoder for FakeDecoder {
    fn push_bytes(&mut self, frame: &[u8]) -> Vec<Result<StreamEvent, LlmError>> {
        let result: Result<Vec<StreamEvent>, LlmError> = {
            let block = self.blocks;
            self.blocks += 1;
            Ok(vec![StreamEvent::TextDelta {
                block,
                text: String::from_utf8_lossy(frame).into_owned(),
            }])
        };
        result
            .map(|events| events.into_iter().map(Ok).collect())
            .unwrap_or_else(|error| vec![Err(error)])
    }
    fn finish(&mut self) -> Vec<Result<StreamEvent, LlmError>> {
        let result: Result<Vec<StreamEvent>, LlmError> = Ok(vec![StreamEvent::End {
            inference: Default::default(),
            stop_reason: StopReason::EndTurn,
            usage: Usage::default().into(),
        }]);
        result
            .map(|events| events.into_iter().map(Ok).collect())
            .unwrap_or_else(|error| vec![Err(error)])
    }

    fn usage_report(&self) -> lingxi_llm_client::protocol::UsageReport {
        let usage = { None };
        let complete = { false };
        lingxi_llm_client::protocol::UsageReport {
            state: if usage.is_none() {
                lingxi_llm_client::protocol::UsageState::Missing
            } else if complete {
                lingxi_llm_client::protocol::UsageState::Complete
            } else {
                lingxi_llm_client::protocol::UsageState::Partial
            },
            usage,
        }
    }
}

// --- fixtures --------------------------------------------------------------

/// A provider entry written the way a user writes settings. No code in this
/// crate mentions any of these names (gate 30).
#[allow(clippy::too_many_arguments)]
fn conn(
    profile_name: &str,
    base: &str,
    models: Value,
    group: Option<&str>,
    order: u32,
    billing: &str,
    hidden: bool,
) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": profile_name,
        "base_url": base,
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": models,
        "pricing": { "billingMode": billing },
        "connection": {
            "group": group,
            "order": order,
            "hidden": hidden,
            "failover": { "rateLimit": true, "overloaded": true, "serverError": true, "network": true, "auth": true },
        },
    }))
    .expect("profile fixture parses")
}

/// A standalone profile: its own one-connection group, no failover.
fn solo(profile_name: &str, base: &str, models: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": profile_name,
        "base_url": base,
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": models,
    }))
    .expect("profile fixture parses")
}

fn model(display: &str, wire: &str) -> Value {
    json!([{
        "display_model": display,
        "request_model": wire,
        "billing_model": wire,
        "metadata": { "contextWindowTokens": 128000 },
    }])
}

fn client(
    profiles: &[ProviderProfile],
    http: Arc<ScriptedTransport>,
) -> lingxi_llm_client::LlmClient {
    let mut b = LlmClientBuilder::with_transport(http, profiles);
    b.register_codec(Arc::new(FakeCodec));
    b.with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .expect("every profile's protocol has a codec")
}

fn failover_pair_with_transport(http: Arc<dyn Transport>) -> lingxi_llm_client::LlmClient {
    let profiles = [
        conn(
            "acme:one",
            "https://one.test",
            model("m-1", "m-1"),
            Some("acme"),
            0,
            "per_token",
            false,
        ),
        conn(
            "acme:two",
            "https://two.test",
            model("m-1", "m-1"),
            Some("acme"),
            1,
            "per_token",
            false,
        ),
    ];
    LlmClientBuilder::with_transport(http, &profiles)
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .expect("built-in OpenAI chat codec is registered")
}

fn openai_openrouter_client(
    http: Arc<ScriptedTransport>,
) -> (lingxi_llm_client::LlmClient, Vec<ProviderProfile>) {
    let mut profiles: Vec<_> = builtin_providers()
        .expect("built-in provider presets parse")
        .into_iter()
        .filter(|p| matches!(p.profile_name.as_str(), "openai" | "openrouter"))
        .collect();
    // The scripted codec models the common chat surface for the test. OpenAI's
    // built-in profile normally uses Responses, while its model names and auth
    // strategy remain the real built-in definitions.
    profiles
        .iter_mut()
        .find(|p| p.profile_name == "openai")
        .expect("the OpenAI preset is present")
        .protocol = ProtocolFamily::OpenAiChat;
    let mut b = LlmClientBuilder::with_transport(http, &profiles);
    b.register_codec(Arc::new(FakeCodec));
    (
        b.with_region(lingxi_llm_client::protocol::Region::International)
            .build()
            .expect("built-in profiles have codecs"),
        profiles,
    )
}

fn request(model: &str) -> CompletionRequest {
    CompletionRequest {
        service_tier: None,
        model: model.to_owned(),
        web_search: None,
        file_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "hi".to_owned(),
                thought_signature: None,
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::auto(),
        max_tokens: None,
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}

fn openai_file_profile(profile_name: &str, order: u32, grouped: bool) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "openai",
        "profile_name": profile_name,
        "base_url": "https://api.openai.com/v1",
        "protocol": "open_ai_responses",
        "auth": "api_key",
        "models": [{
            "display_model": "gpt-test",
            "request_model": "gpt-test",
            "billing_model": "gpt-test",
            "metadata": {"input_modalities": ["text", "image", "file"]},
            "capability_support": {
                "vision": "supported",
                "documents": "supported",
                "tools": "unknown",
                "reasoning": "unknown",
                "signed_reasoning": "unknown",
                "streaming": "unknown",
                "structured_output": "unknown"
            }
        }],
        "connection": if grouped {
            json!({
                "group": "openai-file-failover",
                "connection_id": profile_name,
                "order": order,
                "failover": {"rateLimit": true, "overloaded": true, "serverError": true, "network": true, "auth": true}
            })
        } else {
            json!({})
        }
    }))
    .expect("OpenAI file profile parses")
}

fn request_with_app_document() -> CompletionRequest {
    let mut request = request("gpt-test");
    request.messages[0].content.push(ContentBlock::Document {
        source: DocumentSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "attachment-1".into(),
                revision: "revision-1".into(),
                filename: "guide.pdf".into(),
                media_type: "application/pdf".into(),
                size_bytes: 3,
            },
        },
        title: None,
    });
    request
}

#[test]
fn file_upload_is_repeated_for_the_fallback_profile_and_uses_its_credential() {
    let http = FileFailoverTransport::new(true);
    let profiles = [
        openai_file_profile("primary", 0, true),
        openai_file_profile("backup", 1, true),
    ];
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &profiles);
    builder.with_attachment_resolver(Arc::new(TestAttachmentResolver(Bytes::from_static(b"pdf"))));
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .expect("built-in Responses codec and API-key auth are available");
    let mut options = RequestOptions {
        credential: Some(lingxi_llm_client::protocol::Secret::new(
            "primary-secret".to_owned(),
        )),
        ..RequestOptions::default()
    };
    options.fallback_credentials.insert(
        "backup".into(),
        lingxi_llm_client::protocol::Secret::new("backup-secret".to_owned()),
    );
    let original = request_with_app_document();

    let response = block_on(client.complete(&original, &options)).unwrap();

    assert_eq!(response.executed_profile.as_deref(), Some("backup"));
    assert_eq!(http.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(http.completions.load(Ordering::SeqCst), 2);
    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].2.as_deref(), Some("Bearer primary-secret"));
    assert_eq!(requests[1].2.as_deref(), Some("Bearer primary-secret"));
    assert_eq!(requests[2].2.as_deref(), Some("Bearer backup-secret"));
    assert_eq!(requests[3].2.as_deref(), Some("Bearer backup-secret"));
    let primary_body: Value = serde_json::from_slice(&requests[1].3).unwrap();
    let fallback_body: Value = serde_json::from_slice(&requests[3].3).unwrap();
    assert_eq!(primary_body["input"][0]["content"][1]["file_id"], "file-1");
    assert_eq!(fallback_body["input"][0]["content"][1]["file_id"], "file-2");
    assert_eq!(original, request_with_app_document());
}

#[test]
fn cached_file_404_invalidates_and_reuploads_once_on_the_same_profile() {
    let http = FileFailoverTransport::missing_first_file();
    let profile = openai_file_profile("openai", 0, false);
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(TestAttachmentResolver(Bytes::from_static(b"pdf"))));
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let options = RequestOptions {
        credential: Some(lingxi_llm_client::protocol::Secret::new(
            "openai-secret".to_owned(),
        )),
        file_account_scope: Some("account-1".into()),
        ..RequestOptions::default()
    };

    let response = block_on(client.complete(&request_with_app_document(), &options)).unwrap();

    assert_eq!(response.executed_profile.as_deref(), Some("openai"));
    assert_eq!(http.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(http.completions.load(Ordering::SeqCst), 2);
    let requests = http.requests.lock().unwrap();
    let first: Value = serde_json::from_slice(&requests[1].3).unwrap();
    let retried: Value = serde_json::from_slice(&requests[3].3).unwrap();
    assert_eq!(first["input"][0]["content"][1]["file_id"], "file-1");
    assert_eq!(retried["input"][0]["content"][1]["file_id"], "file-2");
}

#[test]
fn concurrent_requests_share_one_upload_for_the_same_scoped_attachment() {
    let http = FileFailoverTransport::new(false);
    let profile = openai_file_profile("openai", 0, false);
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(TestAttachmentResolver(Bytes::from_static(b"pdf"))));
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request = request_with_app_document();
    let options = RequestOptions {
        credential: Some(lingxi_llm_client::protocol::Secret::new(
            "openai-secret".to_owned(),
        )),
        file_account_scope: Some("account-1".into()),
        ..RequestOptions::default()
    };

    let (first, second) = block_on(async {
        futures::join!(
            client.complete(&request, &options),
            client.complete(&request, &options)
        )
    });

    assert!(first.is_ok());
    assert!(second.is_ok());
    assert_eq!(http.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(http.completions.load(Ordering::SeqCst), 2);
    let requests = http.requests.lock().unwrap();
    let completion_bodies = requests
        .iter()
        .filter(|(_, url, _, _)| url.ends_with("/responses"))
        .map(|(_, _, _, body)| serde_json::from_slice::<Value>(body).unwrap())
        .collect::<Vec<_>>();
    assert!(completion_bodies
        .iter()
        .all(|body| body["input"][0]["content"][1]["file_id"] == "file-1"));
}

struct TestAttachmentResolver(Bytes);

#[async_trait]
impl lingxi_llm_client::AttachmentResolver for TestAttachmentResolver {
    async fn resolve(&self, _attachment: &AttachmentRef) -> Result<Bytes, LlmError> {
        Ok(self.0.clone())
    }
}

#[test]
fn small_images_stay_inline_without_provider_uploads() {
    let http = FileFailoverTransport::new(false);
    let profile = openai_file_profile("openai", 0, false);
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(TestAttachmentResolver(Bytes::from_static(b"png"))));
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let mut request = request("gpt-test");
    request.messages[0].content.push(ContentBlock::Image {
        source: ImageSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "image-1".into(),
                revision: "revision-1".into(),
                filename: "small.png".into(),
                media_type: "image/png".into(),
                size_bytes: 3,
            },
        },
    });
    let options = RequestOptions {
        credential: Some(lingxi_llm_client::protocol::Secret::new(
            "openai-secret".to_owned(),
        )),
        ..RequestOptions::default()
    };

    block_on(client.complete(&request, &options)).unwrap();

    assert_eq!(http.uploads.load(Ordering::SeqCst), 0);
    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].3).unwrap();
    assert_eq!(
        body["input"][0]["content"][1]["image_url"],
        "data:image/png;base64,cG5n"
    );
}

#[test]
fn provider_upload_cache_is_partitioned_by_account_scope() {
    let http = FileFailoverTransport::new(false);
    let profile = openai_file_profile("openai", 0, false);
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(TestAttachmentResolver(Bytes::from_static(b"pdf"))));
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request = request_with_app_document();
    let options = |scope: &str| RequestOptions {
        credential: Some(lingxi_llm_client::protocol::Secret::new(
            "openai-secret".to_owned(),
        )),
        file_account_scope: Some(scope.to_owned()),
        ..RequestOptions::default()
    };

    block_on(async {
        client
            .complete(&request, &options("account-a"))
            .await
            .unwrap();
        client
            .complete(&request, &options("account-b"))
            .await
            .unwrap();
    });

    assert_eq!(http.uploads.load(Ordering::SeqCst), 2);
    let requests = http.requests.lock().unwrap();
    let file_ids = requests
        .iter()
        .filter(|(_, url, _, _)| url.ends_with("/responses"))
        .map(|(_, _, _, body)| {
            serde_json::from_slice::<Value>(body).unwrap()["input"][0]["content"][1]["file_id"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(file_ids, vec!["file-1", "file-2"]);
}

#[test]
fn grok_responses_profile_exposes_document_file_refs_without_changing_chat_profile() {
    let profiles = builtin_providers().unwrap();
    let chat = profiles
        .iter()
        .find(|profile| profile.profile_name == "grok")
        .unwrap();
    let responses = profiles
        .iter()
        .find(|profile| profile.profile_name == "grok-responses")
        .expect("the xAI Responses profile is built in");

    assert_eq!(chat.protocol, ProtocolFamily::OpenAiChat);
    assert_eq!(responses.protocol, ProtocolFamily::OpenAiResponses);
    assert_eq!(
        responses.model_list.shape(responses.protocol),
        Some(ProtocolFamily::OpenAiChat)
    );
    assert_eq!(
        lingxi_llm_client::files::capabilities(chat, "grok-4.20", "application/pdf").model_input,
        lingxi_llm_client::files::ModelFileReference::Unsupported
    );
    assert_eq!(
        lingxi_llm_client::files::capabilities(responses, "grok-4.20", "application/pdf")
            .model_input,
        lingxi_llm_client::files::ModelFileReference::FileId
    );
}

// --- resolve ---------------------------------------------------------------

#[test]
fn siblings_of_one_group_are_the_chain_in_connection_order() {
    let c = client(
        &[
            conn(
                "acme:intl",
                "https://intl.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
            conn(
                "acme:cn",
                "https://cn.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    let route = c.resolve("m-1").unwrap();

    assert_eq!(route.profile_name, "acme:cn", "lowest order starts");
    assert_eq!(
        route
            .connection_chain
            .iter()
            .map(|h| h.profile_name.as_str())
            .collect::<Vec<_>>(),
        vec!["acme:intl"],
        "the rest of the group is the chain, and the head is not in it"
    );
}

#[test]
fn scoping_to_one_connection_still_offers_the_whole_group() {
    let c = client(
        &[
            conn(
                "acme:cn",
                "https://cn.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:intl",
                "https://intl.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    // A picker hands back a connection, so a session stores `acme:intl` and
    // every later request arrives scoped to it. Scoping picks where to start;
    // it does not decide whether the rest of the group may be used.
    let route = c.resolve_in("m-1", Some("acme:intl")).unwrap();

    assert_eq!(route.profile_name, "acme:intl");
    assert_eq!(
        route.connection_chain.len(),
        1,
        "the chain comes from the head's group, not from what was in scope"
    );
}

#[test]
fn a_hop_must_serve_the_same_wire_model() {
    let c = client(
        &[
            conn(
                "acme:a",
                "https://a.test",
                model("m-1", "wire-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:b",
                "https://b.test",
                model("m-1", "wire-2"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    let route = c.resolve("m-1").unwrap();

    assert_eq!(route.request_model, "wire-1");
    assert!(
        route.connection_chain.is_empty(),
        "failing over re-points the endpoint, not the model: a sibling serving a \
         different wire id would silently answer as something else"
    );
}

#[test]
fn a_hop_must_be_billed_the_same_way() {
    let c = client(
        &[
            conn(
                "zhipu:plan",
                "https://plan.test",
                model("glm", "glm"),
                Some("zhipu"),
                0,
                "subscription",
                false,
            ),
            conn(
                "zhipu:metered",
                "https://metered.test",
                model("glm", "glm"),
                Some("zhipu"),
                1,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    let route = c.resolve("glm").unwrap();

    assert_eq!(route.profile_name, "zhipu:plan");
    assert!(
        route.connection_chain.is_empty(),
        "moving a rate-limited subscription request onto the metered endpoint \
         would start charging real money for what the plan already covers"
    );
}

#[test]
fn a_hidden_connection_is_reachable_by_failover_but_never_offered() {
    let c = client(
        &[
            conn(
                "acme:key1",
                "https://one.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:key2",
                "https://two.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                true,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    assert_eq!(
        c.models().iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        vec!["m-1"],
        "the spare key slot is not something a user can pick"
    );
    assert_eq!(
        c.resolve("m-1").unwrap().connection_chain.len(),
        1,
        "but it is still there to fail over onto"
    );
}

#[test]
fn matching_across_two_groups_is_a_real_ambiguity() {
    let c = client(
        &[
            solo("openai", "https://a.test", model("gpt", "gpt")),
            solo("azure", "https://b.test", model("gpt", "gpt")),
        ],
        ScriptedTransport::new(vec![]),
    );

    match c.resolve("gpt").unwrap_err() {
        ResolveError::AmbiguousAcrossGroups { groups, .. } => {
            assert_eq!(groups, vec!["azure".to_owned(), "openai".to_owned()]);
        }
        other => panic!("expected an ambiguity, got {other:?}"),
    }

    // The qualified form a picker shows routes.
    let route = c.resolve("azure/gpt").unwrap();
    assert_eq!(route.profile_name, "azure");
}

#[test]
fn native_slash_ids_and_qualified_refs_require_scope_when_both_match() {
    let http = ScriptedTransport::new(vec![]);
    let (c, _) = openai_openrouter_client(http);

    let err = c
        .resolve("openai/gpt-4o")
        .expect_err("the OpenRouter native id collides with OpenAI/gpt-4o");
    assert!(
        err.to_string().contains("ambiguous"),
        "the caller should be told to scope the conflicting reference: {err}"
    );

    let openai = c
        .resolve_in("openai/gpt-4o", Some("openai"))
        .expect("a connection scope selects the qualified OpenAI model");
    assert_eq!(openai.profile_name, "openai");
    assert_eq!(openai.request_model, "gpt-4o");

    let openrouter = c
        .resolve_in("openai/gpt-4o", Some("openrouter"))
        .expect("a connection scope selects OpenRouter's native slash id");
    assert_eq!(openrouter.profile_name, "openrouter");
    assert_eq!(openrouter.request_model, "openai/gpt-4o");

    let unambiguous_native = c
        .resolve("openrouter/auto")
        .expect("a slash-bearing native id with no qualified collision stays valid");
    assert_eq!(unambiguous_native.profile_name, "openrouter");
    assert_eq!(unambiguous_native.request_model, "openrouter/auto");
}

#[test]
fn native_qualified_collision_detection_compares_the_resolved_head() {
    let same_model = solo(
        "foo",
        "https://same.test",
        json!([{
            "display_model": "foo/bar",
            "request_model": "foo/bar",
            "billing_model": "foo/bar",
            "aliases": ["bar"],
        }]),
    );
    let same_target = client(&[same_model], ScriptedTransport::new(vec![]));
    assert_eq!(
        same_target.resolve("foo/bar").unwrap().request_model,
        "foo/bar",
        "both interpretations reaching the same model entry are unambiguous"
    );

    let one_profile = solo(
        "foo",
        "https://same.test",
        json!([
            {"display_model": "foo/bar", "request_model": "foo/bar", "billing_model": "foo/bar"},
            {"display_model": "bar", "request_model": "wire-bar", "billing_model": "wire-bar"},
        ]),
    );
    let different_model = client(&[one_profile], ScriptedTransport::new(vec![]));
    assert!(matches!(
        different_model.resolve("foo/bar"),
        Err(ResolveError::AmbiguousNativeAndQualified { .. })
    ));

    let mut native = conn(
        "foo:one",
        "https://one.test",
        json!([{"display_model": "foo/bar", "request_model": "foo/bar", "billing_model": "foo/bar"}]),
        Some("foo"),
        0,
        "per_token",
        false,
    );
    native.provider_id = lingxi_llm_client::protocol::ProviderId::new("foo");
    let mut qualified = conn(
        "foo:two",
        "https://two.test",
        json!([{"display_model": "bar", "request_model": "wire-bar", "billing_model": "wire-bar"}]),
        Some("foo"),
        1,
        "per_token",
        false,
    );
    qualified.provider_id = lingxi_llm_client::protocol::ProviderId::new("foo");
    let different_connection = client(&[native, qualified], ScriptedTransport::new(vec![]));
    assert!(matches!(
        different_connection.resolve("foo/bar"),
        Err(ResolveError::AmbiguousNativeAndQualified { .. })
    ));
}

#[test]
fn scoped_completion_stream_and_search_use_the_selected_route_and_credential() {
    let profiles = builtin_providers()
        .expect("built-in provider presets parse")
        .into_iter()
        .filter(|p| matches!(p.profile_name.as_str(), "openai" | "openrouter"))
        .collect::<Vec<_>>();
    let openai_base = profiles
        .iter()
        .find(|p| p.profile_name == "openai")
        .unwrap()
        .base_url
        .clone();
    let openrouter_base = profiles
        .iter()
        .find(|p| p.profile_name == "openrouter")
        .unwrap()
        .base_url
        .clone();
    let http = ScriptedTransport::from_owned(vec![
        (openai_base.clone(), Ok(200)),
        (openrouter_base.clone(), Ok(200)),
    ]);
    let (c, _) = openai_openrouter_client(http.clone());
    let req = request("openai/gpt-4o");
    let with_credential = |secret: &str| RequestOptions {
        credential: Some(lingxi_llm_client::protocol::Secret::new(secret.to_owned())),
        ..RequestOptions::default()
    };

    let (complete, stream, searched, searched_stream) = block_on(async {
        let complete = c
            .complete_in("openai", &req, &with_credential("openai-secret"))
            .await
            .unwrap();
        let stream = c
            .stream_in("openrouter", &req, &with_credential("openrouter-secret"))
            .await
            .unwrap();
        let searched = c
            .web_search_in(
                "openai",
                &req,
                WebSearchConfig {
                    allowed_domains: vec!["example.com".to_owned()],
                    ..Default::default()
                },
                &with_credential("openai-secret"),
            )
            .await
            .unwrap();
        let searched_stream = c
            .web_search_stream_in(
                "openrouter",
                &req,
                WebSearchConfig {
                    blocked_domains: vec!["blocked.example".to_owned()],
                    ..Default::default()
                },
                &with_credential("openrouter-secret"),
            )
            .await
            .unwrap();
        (complete, stream, searched, searched_stream)
    });

    assert_eq!(complete.executed_profile.as_deref(), Some("openai"));
    assert_eq!(stream.executed_profile(), "openrouter");
    assert_eq!(searched.executed_profile.as_deref(), Some("openai"));
    assert_eq!(searched_stream.executed_profile(), "openrouter");
    assert_eq!(
        http.hops(),
        vec![
            format!("{}/chat", openai_base.trim_end_matches('/')),
            format!("{}/chat", openrouter_base.trim_end_matches('/')),
            format!("{}/chat", openai_base.trim_end_matches('/')),
            format!("{}/chat", openrouter_base.trim_end_matches('/')),
        ],
        "each scoped helper must use the endpoint selected by its profile"
    );
    assert_eq!(
        http.auth_headers(),
        vec![
            Some("Bearer openai-secret".to_owned()),
            Some("Bearer openrouter-secret".to_owned()),
            Some("Bearer openai-secret".to_owned()),
            Some("Bearer openrouter-secret".to_owned()),
        ],
        "a key must never follow a qualified-looking model string to another provider"
    );
    let bodies = http
        .bodies()
        .into_iter()
        .map(|body| serde_json::from_slice::<Value>(&body).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(bodies[0]["model"], "gpt-4o");
    assert_eq!(bodies[1]["model"], "openai/gpt-4o");
    assert_eq!(bodies[2]["web_search"]["allowed_domains"][0], "example.com");
    assert_eq!(
        bodies[3]["web_search"]["blocked_domains"][0],
        "blocked.example"
    );
}

#[test]
fn unscoped_collision_fails_before_authentication_or_http() {
    let http = ScriptedTransport::new(vec![]);
    let (c, _) = openai_openrouter_client(http.clone());
    let err = block_on(c.complete(
        &request("openai/gpt-4o"),
        &RequestOptions {
            credential: Some(lingxi_llm_client::protocol::Secret::new(
                "primary-secret".to_owned(),
            )),
            ..RequestOptions::default()
        },
    ))
    .unwrap_err();

    assert!(
        matches!(err, LlmError::ModelUnavailable { ref message } if message.contains("ambiguous")),
        "an unsafe unscoped reference should fail as model resolution: {err}"
    );
    assert!(
        http.hops().is_empty(),
        "no provider should receive the request"
    );
    assert!(
        http.auth_headers().is_empty(),
        "no credential should be applied"
    );
}

#[test]
fn two_models_of_one_profile_answering_to_one_name_is_an_error() {
    let c = client(
        &[solo(
            "acme",
            "https://a.test",
            json!([
                {"display_model": "a", "request_model": "wire-a", "billing_model": "wire-a", "aliases": ["dup"]},
                {"display_model": "b", "request_model": "wire-b", "billing_model": "wire-b", "aliases": ["dup"]},
            ]),
        )],
        ScriptedTransport::new(vec![]),
    );

    assert_eq!(
        c.resolve("dup").unwrap_err(),
        ResolveError::DuplicateOnProfile {
            model: "dup".to_owned(),
            profile_name: "acme".to_owned(),
        },
        "two models on one endpoint: which to send is genuinely unknown"
    );
}

#[test]
fn an_unknown_model_says_so() {
    let c = client(
        &[solo("acme", "https://a.test", model("m-1", "m-1"))],
        ScriptedTransport::new(vec![]),
    );
    assert_eq!(
        c.resolve("nope").unwrap_err(),
        ResolveError::UnknownModel {
            model: "nope".to_owned()
        }
    );
}

// --- gate 32: the walk -----------------------------------------------------

/// The same two connections as `pair`, on the wire that keeps turn state at
/// the endpoint, with the continuation opt-in the encoder requires.
fn stateful_pair(http: Arc<ScriptedTransport>) -> lingxi_llm_client::LlmClient {
    let responses = |name: &str, base: &str, order: u32| -> ProviderProfile {
        let mut p = conn(
            name,
            base,
            model("m-1", "m-1"),
            Some("acme"),
            order,
            "per_token",
            false,
        );
        p.protocol = ProtocolFamily::OpenAiResponses;
        p.extra = json!({"supports_previous_response_id": true});
        p
    };
    let profiles = [
        responses("acme:one", "https://one.test", 0),
        responses("acme:two", "https://two.test", 1),
    ];
    let mut b = LlmClientBuilder::with_transport(http, &profiles);
    b.register_codec(Arc::new(FakeStatefulCodec));
    b.with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .expect("every profile's protocol has a codec")
}

fn continuing(model: &str) -> CompletionRequest {
    let mut req = request(model);
    req.previous_response_id = Some(ResponseId::new("resp_previous"));
    req
}

/// A continuation id is state the *endpoint* holds, so the second connection
/// has never heard of it. Failing over would send a turn that silently loses
/// everything before it — the one failure a retry is supposed to prevent.
///
/// Stated against the same script as the plain failover test above: the only
/// difference between the two is the continuation, so this cannot pass by the
/// chain being empty.
#[test]
fn a_continuation_does_not_fail_over_to_a_connection_that_never_saw_it() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = stateful_pair(http.clone());

    let err = block_on(c.complete(&continuing("m-1"), &RequestOptions::default()))
        .expect_err("the connection holding the state is the only one that can serve this");
    assert!(matches!(err, LlmError::Overloaded { .. }), "{err}");
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat"],
        "the sibling was never tried; it does not have the previous response"
    );

    // The same script, without the continuation, walks both — so the stop
    // above is the continuation and not the route.
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = stateful_pair(http.clone());
    block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap();
    assert_eq!(http.hops().len(), 2);
}

/// A wire with no endpoint-side turn state cannot continue one. Refusing at
/// the client is the layer that matters: a codec that never looks at the field
/// would send a fresh turn, and the caller would be told nothing.
#[test]
fn a_continuation_on_a_wire_that_has_no_such_state_is_refused_before_it_is_sent() {
    let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
    let c = pair(http.clone());

    let err = block_on(c.complete(&continuing("m-1"), &RequestOptions::default()))
        .expect_err("this wire keeps no turn state to continue");
    match err {
        LlmError::UnsupportedCapability { message } => assert!(
            message.contains("acme:one"),
            "the refusal must name the connection: {message}"
        ),
        other => panic!("{other}"),
    }
    assert!(
        http.hops().is_empty(),
        "nothing may go on the wire before the refusal"
    );
}

fn pair(http: Arc<ScriptedTransport>) -> lingxi_llm_client::LlmClient {
    client(
        &[
            conn(
                "acme:one",
                "https://one.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:two",
                "https://two.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
        ],
        http,
    )
}

#[test]
fn a_failover_trigger_moves_the_request_to_the_next_connection() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = pair(http.clone());

    let resp = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap();

    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat", "https://two.test/chat"],
        "both connections were tried, in order"
    );
}

#[test]
fn an_error_another_endpoint_would_repeat_stops_at_the_first_connection() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::ContextOverflow {
                message: "too long".to_owned(),
                limit: None,
                actual: None,
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = pair(http.clone());

    let err = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap_err();

    assert!(matches!(err, LlmError::ContextOverflow { .. }));
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat"],
        "a transcript that does not fit is not an endpoint problem; retrying it \
         elsewhere would only bury the reason"
    );
}

#[test]
fn the_last_error_survives_when_the_group_is_spent() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "one busy".to_owned(),
            }),
        ),
        (
            "https://two.test",
            Err(LlmError::ProviderInternal {
                message: "two broke".to_owned(),
            }),
        ),
    ]);
    let c = pair(http.clone());

    let err = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap_err();

    assert_eq!(
        err,
        LlmError::ProviderInternal {
            message: "two broke".to_owned()
        },
        "the caller sees why the last connection failed, not the first"
    );
    assert_eq!(http.hops().len(), 2);
}

#[test]
fn streaming_walks_the_same_connections_and_decodes_through_the_codec() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::TransportTimeout {
                message: "no answer".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = pair(http.clone());

    let events = block_on(async {
        let mut s = c
            .stream(
                &request("m-1"),
                &RequestOptions {
                    ..RequestOptions::default()
                },
            )
            .await
            .unwrap();
        let mut out = vec![];
        while let Some(ev) = s.next().await {
            out.push(ev.unwrap());
        }
        out
    });

    assert_eq!(http.hops().len(), 2, "the stream failed over too");
    assert_eq!(
        events.len(),
        3,
        "two frames decoded to two deltas, and `finish` emitted the End"
    );
    assert!(matches!(events[0], StreamEvent::TextDelta { block: 0, .. }));
    assert!(matches!(events[2], StreamEvent::End { .. }));
}

#[test]
fn an_interrupted_http_error_body_keeps_the_known_status_and_fails_over() {
    let http = InterruptedErrorBodyTransport::new(true);
    let c = failover_pair_with_transport(http.clone());

    let stream = block_on(c.stream(
        &request("m-1"),
        &RequestOptions {
            ..RequestOptions::default()
        },
    ))
    .expect("a known 429 remains eligible for failover when its body is interrupted");

    assert_eq!(stream.status(), 200);
    assert_eq!(
        http.hops(),
        vec![
            "https://one.test/chat/completions",
            "https://two.test/chat/completions"
        ]
    );
}

#[test]
fn an_exhausted_interrupted_http_error_body_keeps_429_retry_after_and_body() {
    let http = InterruptedErrorBodyTransport::new(false);
    let c = failover_pair_with_transport(http.clone());

    let result = block_on(c.stream(
        &request("m-1"),
        &RequestOptions {
            ..RequestOptions::default()
        },
    ));
    let err = match result {
        Ok(_) => panic!("both 429 responses should remain failures"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::RateLimited { ref message, retry_after: Some(duration) }
            if duration == std::time::Duration::from_secs(7)
                && message.contains("quota exceeded")),
        "the final classified error retains the status, collected body, and Retry-After: {err}"
    );
    assert_eq!(http.hops().len(), 2);
}

#[test]
fn a_standalone_profile_is_one_connection() {
    let http = ScriptedTransport::new(vec![(
        "https://one.test",
        Err(LlmError::Overloaded {
            message: "busy".to_owned(),
        }),
    )]);
    let c = client(
        &[solo("only", "https://one.test", model("m-1", "m-1"))],
        http.clone(),
    );

    let err = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap_err();

    assert!(matches!(err, LlmError::Overloaded { .. }));
    assert_eq!(
        http.hops().len(),
        1,
        "a trigger with nowhere to go is still just the error"
    );
}

/// The streaming path used to return frames and nothing else, so every response
/// header was unreachable on the one path an agent turn actually takes — a
/// streamed 429 could not even say how long to wait. `open_stream` carries the
/// status and headers now; this is what that buys.
#[test]
fn a_streamed_response_still_has_its_headers() {
    let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
    let c = pair(http.clone());

    let s = block_on(async {
        c.stream(
            &request("m-1"),
            &RequestOptions {
                ..RequestOptions::default()
            },
        )
        .await
        .unwrap()
    });

    assert_eq!(s.status(), 200);
    assert_eq!(
        s.header("anthropic-ratelimit-unified-5h-utilization"),
        Some("0.42"),
        "header lookup is case-insensitive; the provider sent it capitalized"
    );
    assert_eq!(
        s.header("Retry-After"),
        Some("7"),
        "the header a streamed 429 needs, and could not reach before"
    );
    assert_eq!(s.header("x-absent"), None);
    assert_eq!(s.headers().len(), 2);
}

/// This crate does not hold, fetch or store credentials: the caller owns them
/// and hands one over per request. There is no credential store to reach into
/// and no `CredentialProvider` to consult — whatever the caller puts on
/// `RequestOptions` is exactly what the authenticator is given.
mod credentials_come_from_the_caller {
    use super::*;
    use std::sync::Mutex;

    /// Records the credential it was handed, so a test can prove which one
    /// arrived rather than that *some* request went out.
    struct Recording(Arc<Mutex<Vec<Option<String>>>>);

    #[async_trait]
    impl lingxi_llm_client::Authenticator for Recording {
        async fn apply(
            &self,
            _req: &mut HttpRequest,
            _profile: &ProviderProfile,
            credential: Option<&lingxi_llm_client::protocol::Secret<String>>,
        ) -> Result<(), LlmError> {
            self.0
                .lock()
                .unwrap()
                .push(credential.map(|s| s.expose_secret().to_owned()));
            Ok(())
        }
    }

    fn keyed_profile() -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id": "acme",
            "profile_name": "acme",
            "base_url": "https://one.test",
            "protocol": "open_ai_chat",
            "auth": "api_key",
            "models": [{"display_model": "m-1", "request_model": "m-1", "billing_model": "m-1"}],
        }))
        .expect("profile fixture parses")
    }

    #[test]
    fn the_authenticator_is_handed_the_caller_s_credential() {
        let seen = Arc::new(Mutex::new(vec![]));
        let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
        let profiles = [keyed_profile()];
        let mut b = LlmClientBuilder::with_transport(http, &profiles);
        b.register_codec(Arc::new(FakeCodec));
        b.register_authenticator(
            lingxi_llm_client::protocol::AuthStrategy::ApiKey,
            Arc::new(Recording(seen.clone())),
        );
        let c = b
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build()
            .expect("the profile's protocol has a codec");

        block_on(async {
            c.complete(
                &request("m-1"),
                &RequestOptions {
                    credential: Some(lingxi_llm_client::protocol::Secret::new(
                        "sk-from-the-caller".to_owned(),
                    )),
                    ..RequestOptions::default()
                },
            )
            .await
            .unwrap();
        });

        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[Some("sk-from-the-caller".to_owned())],
            "the secret the caller passed in, not one this crate went looking for"
        );
    }

    #[test]
    fn a_caller_that_passes_none_leaves_the_authenticator_with_none() {
        // There is nowhere else for one to come from. A crate that fell back to
        // an environment variable or a keychain would be storing credentials.
        let seen = Arc::new(Mutex::new(vec![]));
        let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
        let profiles = [keyed_profile()];
        let mut b = LlmClientBuilder::with_transport(http, &profiles);
        b.register_codec(Arc::new(FakeCodec));
        b.register_authenticator(
            lingxi_llm_client::protocol::AuthStrategy::ApiKey,
            Arc::new(Recording(seen.clone())),
        );
        let c = b
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build()
            .unwrap();

        block_on(async {
            c.complete(&request("m-1"), &RequestOptions::default())
                .await
                .unwrap();
        });

        assert_eq!(seen.lock().unwrap().as_slice(), &[None]);
    }
}

/// Failover re-points the endpoint, not the bill. The connection-level rule
/// already stops a subscription request sliding onto a metered endpoint; an
/// aggregator needs the same rule one level down, because its free tier and its
/// metered models share one endpoint and one key.
#[test]
fn a_free_model_does_not_fail_over_onto_a_metered_one() {
    let free = json!([{
        "display_model": "m-1", "request_model": "m-1", "billing_model": "m-1",
        "billing_mode": "free",
    }]);
    let metered = json!([{
        "display_model": "m-1", "request_model": "m-1", "billing_model": "m-1",
    }]);
    let profiles = [
        conn(
            "a",
            "https://one.test",
            free.clone(),
            Some("g"),
            0,
            "per_token",
            false,
        ),
        conn(
            "b",
            "https://two.test",
            metered,
            Some("g"),
            1,
            "per_token",
            false,
        ),
    ];
    let c = client(&profiles, ScriptedTransport::new(vec![]));
    let route = c.resolve("m-1").expect("the free model resolves");
    assert!(
        route.connection_chain.is_empty(),
        "the metered sibling serves the same wire model over the same key, and \
         falling onto it would start charging: {:?}",
        route.connection_chain
    );

    // Two free connections are interchangeable, which is what makes the check
    // above a billing rule rather than a ban on failover.
    let both_free = [
        conn(
            "a",
            "https://one.test",
            free.clone(),
            Some("g"),
            0,
            "per_token",
            false,
        ),
        conn(
            "b",
            "https://two.test",
            free,
            Some("g"),
            1,
            "per_token",
            false,
        ),
    ];
    let c = client(&both_free, ScriptedTransport::new(vec![]));
    let route = c.resolve("m-1").unwrap();
    assert_eq!(route.connection_chain.len(), 1);
}

/// Production HTTP delivers arbitrary byte chunks rather than parsed events.
mod raw_http_streams {
    use super::*;
    use std::collections::VecDeque;
    use std::time::Duration;

    struct RawTransport {
        responses: Mutex<VecDeque<StreamResponse>>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Transport for RawTransport {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted response"))
        }
    }

    fn response(status: u16, body: &[u8], chunk_size: usize) -> StreamResponse {
        let chunks: Vec<_> = body
            .chunks(chunk_size)
            .map(|b| Ok(Bytes::copy_from_slice(b)))
            .collect();
        StreamResponse {
            status,
            headers: vec![
                (
                    "Content-Type".into(),
                    "Text/Event-Stream; charset=utf-8".into(),
                ),
                ("Retry-After".into(), "7".into()),
            ],
            body: Box::pin(stream::iter(chunks)),
        }
    }

    fn setup(
        responses: Vec<StreamResponse>,
        failover: bool,
    ) -> (lingxi_llm_client::LlmClient, Arc<RawTransport>) {
        let http = Arc::new(RawTransport {
            responses: Mutex::new(responses.into()),
            calls: AtomicUsize::new(0),
        });
        let mut profiles = vec![conn(
            "one",
            "https://one.test",
            model("m-1", "m-1"),
            Some("g"),
            0,
            "per_token",
            false,
        )];
        if failover {
            profiles.push(conn(
                "two",
                "https://two.test",
                model("m-1", "m-1"),
                Some("g"),
                1,
                "per_token",
                false,
            ));
        }
        (
            LlmClientBuilder::with_transport(http.clone(), &profiles)
                .with_region(lingxi_llm_client::protocol::Region::International)
                .build()
                .unwrap(),
            http,
        )
    }

    const SSE: &[u8] = concat!(
        ": keepalive\r\n\r\n",
        "event: message\r\ndata: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你好\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]"
    ).as_bytes();

    #[test]
    fn sse_decodes_at_every_byte_boundary_and_coalesced_events() {
        for chunk_size in [1, 2, 7, SSE.len()] {
            let (client, _) = setup(vec![response(200, SSE, chunk_size)], false);
            block_on(async {
                let mut stream = client
                    .stream(&request("m-1"), &RequestOptions::default())
                    .await
                    .unwrap();
                let mut text = String::new();
                let mut ends = 0;
                while let Some(event) = stream.next().await {
                    match event.unwrap() {
                        StreamEvent::TextDelta { text: delta, .. } => text.push_str(&delta),
                        StreamEvent::End { .. } => ends += 1,
                        _ => {}
                    }
                }
                assert_eq!(text, "你好", "chunk size {chunk_size}");
                assert_eq!(ends, 1);
            });
        }
    }

    #[test]
    fn http_auth_and_rate_limit_statuses_fail_over_before_returning_a_stream() {
        for status in [401, 429] {
            let (client, http) = setup(
                vec![
                    response(status, br#"{"error":{"message":"try next"}}"#, 3),
                    response(200, SSE, 4),
                ],
                true,
            );
            block_on(async {
                let mut stream = client
                    .stream(&request("m-1"), &RequestOptions::default())
                    .await
                    .unwrap();
                assert_eq!(stream.status(), 200);
                while let Some(event) = stream.next().await {
                    event.unwrap();
                }
            });
            assert_eq!(http.calls.load(Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn exhausted_http_errors_keep_the_provider_message_and_retry_after() {
        for status in [401, 429] {
            let (client, _) = setup(
                vec![response(
                    status,
                    br#"{"error":{"message":"quota unavailable"}}"#,
                    2,
                )],
                false,
            );
            let result = block_on(client.stream(&request("m-1"), &RequestOptions::default()));
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("HTTP error returned a stream"),
            };
            match error {
                LlmError::RateLimited {
                    message,
                    retry_after,
                } if status == 429 => {
                    assert!(message.contains("quota unavailable"));
                    assert_eq!(retry_after, Some(Duration::from_secs(7)));
                }
                LlmError::Authentication { message } if status == 401 => {
                    assert!(message.contains("quota unavailable"))
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn oversized_error_body_is_bounded_and_does_not_poll_the_tail() {
        let mut resp = response(429, &vec![b'x'; 64 * 1024], 4096);
        use futures::StreamExt;
        resp.body = Box::pin(resp.body.chain(stream::once(async {
            panic!("error bodies must not be collected past the limit");
            #[allow(unreachable_code)]
            Ok(Bytes::new())
        })));
        let (client, _) = setup(vec![resp], false);
        assert!(matches!(
            block_on(client.stream(&request("m-1"), &RequestOptions::default())),
            Err(LlmError::RateLimited { .. })
        ));
    }
}

#[test]
fn qualified_profile_wins_over_its_namesake_group() {
    let c = client(
        &[
            conn(
                "acme",
                "https://primary.test",
                model("m", "primary-model"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
            conn(
                "spare",
                "https://spare.test",
                model("m", "other-model"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );
    for scope in [None, Some("acme")] {
        let route = c.resolve_in("acme/m", scope).unwrap();
        assert_eq!(route.profile_name, "acme");
        assert_eq!(route.request_model, "primary-model");
    }
}

struct ModeCheckingCodec(bool);

#[async_trait]
impl WireCodec for ModeCheckingCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
    }

    fn encode_request(
        &self,
        input: lingxi_llm_client::EncodeRequest<'_>,
        context: &lingxi_llm_client::CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let req = input.request();
        let profile = context.profile();
        let route = &wire_api::route(context);
        let opts = &wire_api::options(context);

        assert_eq!(
            opts.stream, self.0,
            "client method must determine wire mode"
        );
        FakeCodec.encode_request(
            lingxi_llm_client::EncodeRequest::new(req),
            &wire_api::context(profile, &(route).request_model, opts),
        )
    }
    fn decode_response(
        &self,
        resp: &HttpResponse,
        _context: &lingxi_llm_client::CodecContext,
    ) -> Result<CompletionResponse, LlmError> {
        FakeCodec.decode_response(resp, &wire_api::decode_context())
    }
    fn stream_decoder(&self, _context: &lingxi_llm_client::CodecContext) -> Box<dyn StreamDecoder> {
        FakeCodec.stream_decoder(&wire_api::decode_context())
    }
}

#[test]
fn client_methods_determine_wire_mode() {
    for streaming in [false, true] {
        let http = ScriptedTransport::new(vec![("https://x.test", Ok(200))]);
        let mut builder = LlmClientBuilder::with_transport(
            http,
            &[solo("acme", "https://x.test", model("m", "m"))],
        );
        builder.register_codec(Arc::new(ModeCheckingCodec(streaming)));
        let client = builder
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build()
            .unwrap();
        let opts = RequestOptions {
            ..Default::default()
        };
        if streaming {
            assert!(block_on(client.stream(&request("m"), &opts)).is_ok());
        } else {
            assert!(block_on(client.complete(&request("m"), &opts)).is_ok());
        }
    }
}

#[derive(Default)]
struct BodyRecorder {
    bodies: Mutex<Vec<Vec<u8>>>,
}

#[async_trait]
impl Transport for BodyRecorder {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl BodyRecorder {
    async fn response(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.bodies.lock().unwrap().push(request.body.to_vec());
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: Bytes::from_static(
                br#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"model":"m"}"#,
            ),
        })
    }
}

#[test]
fn complete_cannot_be_changed_to_streaming_by_profile_body_extras() {
    let mut profile = solo("acme", "https://a.test", model("m", "m"));
    profile.extra = json!({"body": {"stream": true}});
    let http = Arc::new(BodyRecorder::default());
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();

    block_on(client.complete(
        &request("m"),
        &RequestOptions {
            ..RequestOptions::default()
        },
    ))
    .expect("the completion response should decode normally");

    let bodies = http.bodies.lock().unwrap();
    let body: Value = serde_json::from_slice(&bodies[0]).unwrap();
    assert_eq!(body.get("stream"), None);
}

#[test]
fn a_fallback_without_its_own_credential_is_never_sent() {
    use lingxi_llm_client::protocol::Secret;

    let mut primary = conn(
        "primary",
        "https://one.test",
        model("m", "m"),
        Some("g"),
        0,
        "per_token",
        false,
    );
    let mut secondary = conn(
        "secondary",
        "https://two.test",
        model("m", "m"),
        Some("g"),
        1,
        "per_token",
        true,
    );
    primary.auth = lingxi_llm_client::protocol::AuthStrategy::ApiKey;
    secondary.auth = lingxi_llm_client::protocol::AuthStrategy::ApiKey;
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".into(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let client = client(&[primary, secondary], http.clone());
    let opts = RequestOptions {
        credential: Some(Secret::new("primary-secret".to_owned())),
        ..Default::default()
    };
    assert!(block_on(client.complete(&request("m"), &opts)).is_err());
    assert_eq!(http.hops(), vec!["https://one.test/chat"]);
}

#[test]
fn a_fallback_uses_its_explicit_credential_and_reports_the_actual_profile() {
    use lingxi_llm_client::protocol::{Secret, Submission, TokenPricing};
    use std::collections::BTreeMap;

    let mut primary = conn(
        "primary",
        "https://one.test",
        model("m", "m"),
        Some("g"),
        0,
        "per_token",
        false,
    );
    let mut secondary = conn(
        "secondary",
        "https://two.test",
        model("m", "m"),
        Some("g"),
        1,
        "per_token",
        true,
    );
    primary.auth = lingxi_llm_client::protocol::AuthStrategy::ApiKey;
    secondary.auth = lingxi_llm_client::protocol::AuthStrategy::ApiKey;
    primary.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(1.0),
        ..Default::default()
    });
    secondary.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(2.0),
        ..Default::default()
    });
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".into(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let client = client(&[primary, secondary], http.clone());
    let opts = RequestOptions {
        credential: Some(Secret::new("primary-secret".to_owned())),
        fallback_credentials: BTreeMap::from([(
            "secondary".to_owned(),
            Secret::new("secondary-secret".to_owned()),
        )]),
        ..Default::default()
    };
    let mut response = block_on(client.complete(&request("m"), &opts)).unwrap();
    assert_eq!(response.executed_profile.as_deref(), Some("secondary"));
    let route = client.resolve("m").unwrap();
    let usage = Usage {
        input_tokens: 1_000_000,
        ..Default::default()
    };
    assert_eq!(
        client
            .estimate_cost(&route, &usage, &PricingContext::default())
            .unwrap()
            .total_cost,
        1.0
    );
    response.usage = lingxi_llm_client::protocol::UsageReport::measured(
        usage,
        lingxi_llm_client::protocol::UsageState::Complete,
    );
    assert_eq!(
        client
            .estimate_actual_cost(&route, &response, Submission::Interactive)
            .unwrap()
            .total_cost,
        2.0
    );
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat", "https://two.test/chat"]
    );
    assert_eq!(
        http.auth_headers(),
        vec![
            Some("Bearer primary-secret".into()),
            Some("Bearer secondary-secret".into())
        ]
    );
    let stream = block_on(client.stream(&request("m"), &opts)).unwrap();
    assert_eq!(stream.executed_profile(), "secondary");
    assert_eq!(
        client
            .estimate_cost_for_profile(
                &route,
                stream.executed_profile(),
                &response.usage,
                &lingxi_llm_client::protocol::InferenceReport::default(),
                Submission::Interactive
            )
            .unwrap()
            .total_cost,
        2.0
    );
}

#[test]
fn request_timeout_defaults_to_120_seconds_and_can_be_overridden() {
    use std::time::Duration;
    let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
    let client = client(
        &[solo("only", "https://one.test", model("m", "m"))],
        http.clone(),
    );
    block_on(client.complete(&request("m"), &RequestOptions::default())).unwrap();
    block_on(client.complete(
        &request("m"),
        &RequestOptions {
            total_timeout: Some(Duration::from_secs(7)),
            ..Default::default()
        },
    ))
    .unwrap();
    let timeouts = http.timeouts();
    assert!(
        timeouts[0].is_some_and(|t| t <= Duration::from_secs(120) && t > Duration::from_secs(119))
    );
    assert!(timeouts[1].is_some_and(|t| t <= Duration::from_secs(7) && t > Duration::from_secs(6)));
    block_on(client.stream(&request("m"), &RequestOptions::default())).unwrap();
    assert_eq!(
        http.timeouts()[2],
        None,
        "streaming has no default total deadline"
    );
}

#[tokio::test]
async fn total_timeout_expires_during_authentication_before_a_request_is_sent() {
    use lingxi_llm_client::protocol::{AuthStrategy, Secret};
    use std::time::Duration;

    struct SlowAuthenticator;

    #[async_trait]
    impl lingxi_llm_client::Authenticator for SlowAuthenticator {
        async fn apply(
            &self,
            _req: &mut HttpRequest,
            _profile: &ProviderProfile,
            _credential: Option<&Secret<String>>,
        ) -> Result<(), LlmError> {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        }
    }

    for streaming in [false, true] {
        let mut profile = solo("only", "https://one.test", model("m", "m"));
        profile.auth = AuthStrategy::ApiKey;
        let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
        let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
        builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(SlowAuthenticator));
        let client = builder
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build()
            .unwrap();
        let options = RequestOptions {
            total_timeout: Some(Duration::from_millis(10)),
            ..Default::default()
        };

        let result = if streaming {
            client.stream(&request("m"), &options).await.map(|_| ())
        } else {
            client.complete(&request("m"), &options).await.map(|_| ())
        };
        assert!(matches!(result, Err(LlmError::TransportTimeout { .. })));
        assert!(http.hops().is_empty(), "expired work must never be sent");
    }
}
