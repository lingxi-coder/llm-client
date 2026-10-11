use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    audio::*,
    protocol::{LlmError, ProviderProfile, Region, Secret},
    *,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
struct Mock {
    sent: Mutex<Vec<(String, Vec<u8>)>>,
    response: Mutex<Vec<u8>>,
    failure: bool,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, r: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push((r.url, r.body.to_vec()));
        self.reply()
    }
    async fn send_stream(&self, mut r: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let mut body = vec![];
        while let Some(chunk) = r.body.next().await {
            body.extend_from_slice(&chunk?);
        }
        self.sent.lock().unwrap().push((r.url, body));
        self.reply()
    }
}
impl Mock {
    fn reply(&self) -> Result<StreamResponse, LlmError> {
        if self.failure {
            return Err(LlmError::Transport {
                message: "disconnected after upload".into(),
            });
        }
        let bytes = Bytes::from(self.response.lock().unwrap().clone());
        let content_type = if bytes.starts_with(b"data:") {
            "text/event-stream"
        } else if bytes.starts_with(b"{") {
            "application/json"
        } else {
            "audio/pcm"
        };
        Ok(StreamResponse {
            status: 200,
            headers: vec![
                ("content-type".into(), content_type.into()),
                ("x-request-id".into(), "request-1".into()),
            ],
            body: stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}
fn profile(id: &str, name: &str, url: &str) -> ProviderProfile {
    let mut value = json!({"provider_id":id,"profile_name":name,"base_url":url,"protocol":"open_ai_chat","auth":"api_key","models":[{"display_model":"chat-only","request_model":"chat-only","billing_model":"chat-only"}]});
    if id == "openai" {
        value["audio"] = json!({"mode":"enabled","value":{"transcriptions_endpoint":"https://api.openai.com/v1/audio/transcriptions","translations_endpoint":"https://api.openai.com/v1/audio/translations","speech_endpoint":"https://api.openai.com/v1/audio/speech","auth":{"type":"bearer"}}});
    }
    serde_json::from_value(value).unwrap()
}
fn setup(
    profiles: Vec<ProviderProfile>,
    response: Vec<u8>,
    failure: bool,
) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        response: Mutex::new(response),
        failure,
        ..Default::default()
    });
    let mut builder = LlmClientBuilder::with_transport(mock.clone(), &profiles)
        .with_region(Region::International);
    builder.register_authenticator(
        lingxi_llm_client::protocol::AuthStrategy::GcpToken,
        Arc::new(BearerAuthenticator),
    );
    (builder.build().unwrap(), mock)
}
fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("host-credential".into())),
        ..Default::default()
    }
}
fn input() -> AudioInput {
    AudioInput::from_bytes(
        "voice.wav",
        "audio/wav",
        Bytes::from_static(&[0, 255, 1, 2]),
    )
}
async fn collect(output: AudioOutput) -> CollectedAudio {
    match output {
        AudioOutput::Stream(stream) => stream.collect(1024).await.unwrap(),
        AudioOutput::Url(_) => panic!("expected raw output"),
    }
}

#[tokio::test]
async fn audio_defaults_are_independent_of_chat_and_exact_profile() {
    let mut a = profile("openai", "profile-a", "https://api.openai.com/v1");
    a.connection.group = Some("group".into());
    let mut b = a.clone();
    b.profile_name = "profile-b".into();
    let (client, mock) = setup(
        vec![a, b],
        br#"{"text":"hello","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}"#
            .to_vec(),
        false,
    );
    let route = AudioRoute::new("profile-b", "account-b");
    assert!(client
        .audio()
        .capabilities(&AudioRoute::new("group", "account-b"))
        .is_err());
    let caps = client.audio().capabilities(&route).unwrap();
    assert_eq!(caps.profile_name, "profile-b");
    assert_eq!(
        caps.operation(AudioOperation::FileTranscription)
            .unwrap()
            .default_model
            .as_deref(),
        Some("gpt-4o-mini-transcribe")
    );
    let result = client
        .snapshot()
        .audio()
        .transcribe(
            &route,
            input(),
            &TranscriptionRequest::default(),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(result.text, "hello");
    assert_eq!(result.route, route);
    assert_eq!(result.usage.total_tokens, Some(5));
    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    let body = String::from_utf8_lossy(&sent[0].1);
    assert!(body.contains("gpt-4o-mini-transcribe"));
    assert!(!body.contains("chat-only"));
}
#[tokio::test]
async fn unsupported_model_and_operation_fail_without_polling_input_or_transport() {
    let (client, mock) = setup(
        vec![
            profile(
                "qwen",
                "qwen",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            profile("openai", "openai", "https://api.openai.com/v1"),
        ],
        vec![],
        false,
    );
    for (name, model) in [("qwen", None), ("openai", Some("chat-only".into()))] {
        let polled = Arc::new(AtomicUsize::new(0));
        let count = polled.clone();
        let input = AudioInput {
            filename: "voice.wav".into(),
            media_type: "audio/wav".into(),
            size_bytes: 4,
            body: stream::once(async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(Bytes::from_static(b"data"))
            })
            .boxed(),
        };
        let error = client
            .audio()
            .transcribe(
                &AudioRoute::new(name, "account"),
                input,
                &TranscriptionRequest {
                    model,
                    ..Default::default()
                },
                &options(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind, AudioErrorKind::Unsupported);
        assert_eq!(error.dispatch(), AudioDispatch::NotSent);
        assert_eq!(polled.load(Ordering::SeqCst), 0);
    }
    assert!(mock.sent.lock().unwrap().is_empty());
}
#[tokio::test]
async fn scope_mismatch_and_oversize_upload_fail_before_input() {
    let (client, mock) = setup(
        vec![profile("openai", "openai", "https://api.openai.com/v1")],
        vec![],
        false,
    );
    let mut opts = options();
    opts.account_scope = Some("different-account".into());
    assert_eq!(
        client
            .audio()
            .transcribe(
                &AudioRoute::new("openai", "account"),
                input(),
                &TranscriptionRequest::default(),
                &opts
            )
            .await
            .unwrap_err()
            .dispatch(),
        AudioDispatch::NotSent
    );
    let input = AudioInput {
        size_bytes: 25_000_001,
        ..input()
    };
    assert_eq!(
        client
            .audio()
            .transcribe(
                &AudioRoute::new("openai", "account"),
                input,
                &TranscriptionRequest::default(),
                &options()
            )
            .await
            .unwrap_err()
            .kind,
        AudioErrorKind::MediaTooLarge
    );
    assert!(mock.sent.lock().unwrap().is_empty());
}
#[tokio::test]
async fn unknown_dispatch_never_replays_to_group_or_backup() {
    let mut a = profile("openai", "a", "https://api.openai.com/v1");
    a.connection.group = Some("group".into());
    let mut b = a.clone();
    b.profile_name = "b".into();
    let (client, mock) = setup(vec![a, b], vec![], true);
    let error = client
        .audio()
        .transcribe(
            &AudioRoute::new("a", "account"),
            input(),
            &TranscriptionRequest::default(),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), AudioDispatch::Unknown);
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn pcm_remains_binary_and_collection_is_bounded() {
    let bytes = vec![0, 255, 1, 128];
    let (client, mock) = setup(
        vec![profile("openai", "openai", "https://api.openai.com/v1")],
        bytes.clone(),
        false,
    );
    let route = AudioRoute::new("openai", "account");
    let output = collect(
        client
            .audio()
            .synthesize(&route, &SynthesisRequest::new("hello"), &options())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(output.bytes.as_ref(), bytes);
    assert_eq!(output.metadata.format, AudioFormat::Pcm16Le);
    assert_eq!(output.metadata.sample_rate_hz, Some(24000));
    assert_eq!(output.metadata.channels, Some(1));
    let sent = mock.sent.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].1).unwrap();
    assert_eq!(body["model"], "gpt-4o-mini-tts");
    assert_eq!(body["response_format"], "pcm");
    drop(sent);
    let AudioOutput::Stream(stream) = client
        .audio()
        .synthesize(&route, &SynthesisRequest::new("hello"), &options())
        .await
        .unwrap()
    else {
        panic!()
    };
    let error = stream.collect(3).await.unwrap_err();
    assert_eq!(error.kind, AudioErrorKind::MediaTooLarge);
    assert_eq!(error.dispatch(), AudioDispatch::Accepted);
}
#[tokio::test]
async fn xai_raw_transcription_keeps_declared_format_and_default_model() {
    let (client, mock) = setup(
        vec![profile("xai", "xai", "https://api.x.ai/v1")],
        br#"{"text":"raw audio","duration":1.0}"#.to_vec(),
        false,
    );
    let request = TranscriptionRequest {
        raw_format: Some(RawAudioFormat {
            format: AudioFormat::Pcm16Le,
            sample_rate_hz: 16000,
            channels: 1,
        }),
        ..Default::default()
    };
    let raw = AudioInput::from_bytes("voice.pcm", "audio/pcm", Bytes::from_static(b"data"));
    let result = client
        .audio()
        .transcribe(
            &AudioRoute::new("xai", "account"),
            raw,
            &request,
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(result.text, "raw audio");
    let sent = mock.sent.lock().unwrap();
    let body = String::from_utf8_lossy(&sent[0].1);
    assert!(body.contains("grok-voice-transcribe-2.0"));
    assert!(body.contains("16000"));
    assert!(body.contains("pcm"));
}
#[test]
fn restricted_auth_and_glm_international_cannot_gain_cloud_synthesis() {
    let mut oauth = profile("openai", "plan", "https://api.openai.com/v1");
    oauth.auth = lingxi_llm_client::protocol::AuthStrategy::ChatGptPlan;
    oauth.protocol = lingxi_llm_client::protocol::ProtocolFamily::OpenAiResponses;
    oauth.audio = lingxi_llm_client::protocol::ServiceSetting::Disabled;
    oauth.model_list = lingxi_llm_client::protocol::DirectoryRoute::NotPublished;
    oauth.pricing.billing_mode = lingxi_llm_client::protocol::BillingMode::Subscription;
    let (client, _) = setup(
        vec![
            oauth,
            profile("zhipu", "glm-global", "https://api.z.ai/api/paas/v4"),
        ],
        vec![],
        false,
    );
    assert!(!client
        .audio()
        .capabilities(&AudioRoute::new("plan", "account"))
        .unwrap()
        .supports(AudioOperation::Synthesis));
    let glm = client
        .audio()
        .capabilities(&AudioRoute::new("glm-global", "account"))
        .unwrap();
    assert!(glm.supports(AudioOperation::FileTranscription));
    assert!(!glm.supports(AudioOperation::Synthesis));
}
#[tokio::test]
async fn google_and_vertex_use_distinct_models_auth_and_raw_metadata() {
    let mut gemini = profile(
        "google",
        "developer",
        "https://generativelanguage.googleapis.com/v1beta",
    );
    gemini.protocol = lingxi_llm_client::protocol::ProtocolFamily::GeminiGenerateContent;
    let (client,mock)=setup(vec![gemini],serde_json::to_vec(&json!({"id":"interaction-1","model":"gemini-3.8-flash-tts","steps":[{"type":"model_output","content":[{"type":"audio","data":"AP8BgA==","mime_type":"audio/l16","sample_rate":24000}]}],"usage":{"input_tokens":7,"output_tokens":9}})).unwrap(),false);
    let output = collect(
        client
            .audio()
            .synthesize(
                &AudioRoute::new("developer", "account"),
                &SynthesisRequest::new("hello"),
                &options(),
            )
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(output.bytes.as_ref(), [0, 255, 1, 128]);
    assert_eq!(output.usage.input_tokens, Some(7));
    assert!(mock.sent.lock().unwrap()[0].0.ends_with("/interactions"));
    let mut vertex = profile("google", "vertex", "https://aiplatform.googleapis.com");
    vertex.protocol = lingxi_llm_client::protocol::ProtocolFamily::VertexGemini;
    vertex.auth = lingxi_llm_client::protocol::AuthStrategy::GcpToken;
    vertex.signing = Some(lingxi_llm_client::protocol::SigningConfig {
        project: Some("project-id".into()),
        region: Some("global".into()),
        ..Default::default()
    });
    let (client,mock)=setup(vec![vertex],serde_json::to_vec(&json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"AP8BgA=="}}]},"finishReason":"STOP"}],"modelVersion":"gemini-2.5-flash-tts","usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}})).unwrap(),false);
    let output = collect(
        client
            .audio()
            .synthesize(
                &AudioRoute::new("vertex", "account"),
                &SynthesisRequest::new("hello"),
                &options(),
            )
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(output.metadata.model, "gemini-2.5-flash-tts");
    assert_eq!(output.usage.total_tokens, Some(5));
    assert!(mock.sent.lock().unwrap()[0]
        .0
        .contains("projects/project-id/locations/global"));
}
#[tokio::test]
async fn qwen_stream_usage_requires_terminal_completion_and_preserves_pcm() {
    let mut qwen = profile(
        "qwen",
        "qwen",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    );
    qwen.extra = json!({"workspace_id":"workspace-1"});
    let delta = json!({"request_id":"qwen-1","output":{"finish_reason":null,"audio":{"data":"AP8BgA==","id":"audio-1"}}});
    let terminal = json!({"request_id":"qwen-1","output":{"finish_reason":"stop","audio":{"data":"","url":"http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/out.wav?Expires=1&Signature=private","id":"audio-1","expires_at":1800000000}},"usage":{"characters":5}});
    let (client, mock) = setup(
        vec![qwen],
        format!("data: {delta}\n\ndata: {terminal}\n\n").into_bytes(),
        false,
    );
    let output = collect(
        client
            .audio()
            .synthesize(
                &AudioRoute::new("qwen", "account"),
                &SynthesisRequest::new("hello"),
                &options(),
            )
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(output.bytes.as_ref(), [0, 255, 1, 128]);
    assert_eq!(output.usage.characters, Some(5));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}
#[test]
fn openrouter_models_require_explicit_operation_metadata_and_never_chat_modality() {
    let mut p = profile("openrouter", "router", "https://openrouter.ai/api/v1");
    let (client, _) = setup(vec![p.clone()], vec![], false);
    let caps = client
        .audio()
        .capabilities(&AudioRoute::new("router", "account"))
        .unwrap();
    assert!(caps.model(AudioOperation::Synthesis, None).is_err());
    assert!(caps
        .model(AudioOperation::FileTranscription, Some("chat-only"))
        .is_err());
    p.models[0].request_model = "openai/whisper-1".into();
    p.models[0].metadata.output_modalities = vec!["transcription".into()];
    let (client, _) = setup(vec![p], vec![], false);
    let caps = client
        .audio()
        .capabilities(&AudioRoute::new("router", "account"))
        .unwrap();
    assert!(caps
        .model(AudioOperation::FileTranscription, Some("openai/whisper-1"))
        .is_ok());
    assert!(caps
        .model(AudioOperation::Synthesis, Some("openai/whisper-1"))
        .is_err());
}

#[tokio::test]
async fn minimax_normalizes_pcm_and_reported_character_usage() {
    let (client,mock)=setup(vec![profile("minimax","minimax","https://api.minimax.io/v1")],serde_json::to_vec(&json!({"data":{"audio":"00ff0180","status":2},"trace_id":"minimax-1","extra_info":{"audio_sample_rate":24000,"audio_format":"pcm","audio_channel":1,"usage_characters":5},"base_resp":{"status_code":0,"status_msg":"success"}})).unwrap(),false);
    let mut request = SynthesisRequest::new("hello");
    request.voice = Some("discovered-voice".into());
    let output = collect(
        client
            .audio()
            .synthesize(&AudioRoute::new("minimax", "account"), &request, &options())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(output.bytes.as_ref(), [0, 255, 1, 128]);
    assert_eq!(output.usage.characters, Some(5));
    let sent = mock.sent.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].1).unwrap();
    assert_eq!(body["model"], "speech-2.8-hd");
    assert_eq!(body["audio_setting"]["format"], "pcm");
    assert_eq!(body["audio_setting"]["sample_rate"], 24000);
}
#[tokio::test]
async fn glm_common_transcription_is_native_and_region_scoped() {
    let (client, mock) = setup(
        vec![profile("zhipu", "glm", "https://api.z.ai/api/paas/v4")],
        br#"{"text":"hello","model":"glm-asr-2512","id":"glm-1"}"#.to_vec(),
        false,
    );
    let output = client
        .audio()
        .transcribe(
            &AudioRoute::new("glm", "account"),
            input(),
            &TranscriptionRequest::default(),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(output.text, "hello");
    assert_eq!(output.model, "glm-asr-2512");
    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert!(sent[0].0.starts_with("https://api.z.ai/"));
    assert!(String::from_utf8_lossy(&sent[0].1).contains("glm-asr-2512"));
}
#[tokio::test]
async fn explicit_openrouter_speech_preserves_unknown_pcm_geometry() {
    let mut router = profile("openrouter", "router", "https://openrouter.ai/api/v1");
    router.models[0].request_model = "discovered/speech".into();
    router.models[0].metadata.output_modalities = vec!["speech".into()];
    let (client, mock) = setup(vec![router], vec![0, 255, 1, 128], false);
    let mut request = SynthesisRequest::new("hello");
    request.model = Some("discovered/speech".into());
    request.voice = Some("discovered-voice".into());
    let output = collect(
        client
            .audio()
            .synthesize(&AudioRoute::new("router", "account"), &request, &options())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(output.bytes.as_ref(), [0, 255, 1, 128]);
    assert_eq!(output.metadata.sample_rate_hz, None);
    assert_eq!(output.metadata.channels, None);
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}
#[test]
fn native_realtime_catalog_is_independent_of_http_audio_routes() {
    let mut openai = profile("openai", "openai", "https://api.openai.com/v1");
    openai.audio = lingxi_llm_client::protocol::ServiceSetting::Disabled;
    let (client, _) = setup(vec![openai], vec![], false);
    let caps = client
        .audio()
        .capabilities(&AudioRoute::new("openai", "account"))
        .unwrap();
    assert!(!caps.supports(AudioOperation::Synthesis));
    assert!(caps.agent_conversation);
    assert!(caps.native_realtime_contract.unwrap().agent_conversation());
}
