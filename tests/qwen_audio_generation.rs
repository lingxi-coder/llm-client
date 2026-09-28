use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::audio_generation::{
        QwenAudioGenerationChannels, QwenAudioGenerationDispatch, QwenAudioGenerationError,
        QwenAudioGenerationFormat, QwenAudioGenerationReference, QwenAudioGenerationRequest,
        QwenAudioGenerationSampleRate, QwenAudioGenerationScope, QwenAudioGenerationService,
        QwenAudioReferenceFormat, QWEN_AUDIO_GENERATION_PATH,
    },
    HttpRequest, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex, time::Duration};

struct Reply {
    status: u16,
    body: Vec<u8>,
}

#[derive(Clone, Debug)]
struct Sent {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    sent: Mutex<Vec<Sent>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
        }
    }

    fn sent(&self) -> Vec<Sent> {
        self.sent.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(Sent {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected Qwen Audio Generation request")?;
        Ok(StreamResponse {
            status: reply.status,
            headers: Vec::new(),
            body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }
}

fn scope() -> QwenAudioGenerationScope {
    QwenAudioGenerationScope::new("qwen-beijing", "account-7", "workspace-ab12").unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("beijing-key".to_owned())),
        account_scope: Some("account-7".to_owned()),
        ..Default::default()
    }
}

fn reply(status: u16, value: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: serde_json::to_vec(&value).unwrap(),
    })
}

fn success(url: &str) -> Value {
    json!({
        "request_id": "audio-generation-42",
        "output": {
            "finish_reason": "stop",
            "audio": {
                "data": "",
                "url": url,
                "id": "audio-42",
                "expires_at": 1_800_000_000_u64,
                "duration": 3.25
            }
        },
        "usage": { "duration": 3 }
    })
}

#[tokio::test]
async fn sends_native_audio_generation_request_and_returns_unfetched_url() {
    let signed_url = "https://result.example/audio.wav?Expires=123&Signature=private";
    let transport = MockTransport::new([reply(200, success(signed_url))]);
    let service = QwenAudioGenerationService::new(&transport, scope()).unwrap();
    let request =
        QwenAudioGenerationRequest::new("@voice1 says hello; then a soft ambient sound follows.")
            .with_reference(
                QwenAudioGenerationReference::from_bytes(
                    QwenAudioReferenceFormat::Wav,
                    Bytes::from_static(b"RIFF-reference"),
                )
                .unwrap(),
            )
            .with_reference(
                QwenAudioGenerationReference::from_url(
                    "https://public.example/ref.mp3?token=reference-secret",
                )
                .unwrap(),
            )
            .with_format(QwenAudioGenerationFormat::Mp3)
            .with_sample_rate(QwenAudioGenerationSampleRate::Hz44100)
            .with_channels(QwenAudioGenerationChannels::Mono)
            .with_volume(71)
            .with_constant_bitrate(192)
            .with_rate(1.25)
            .with_seed(17)
            .with_aigc_tag(true);

    let result = service.synthesize(&request, &options()).await.unwrap();
    assert_eq!(result.request_id.as_deref(), Some("audio-generation-42"));
    assert_eq!(result.audio_url(), signed_url);
    assert_eq!(result.audio_id.as_deref(), Some("audio-42"));
    assert_eq!(result.expires_at_unix_seconds, Some(1_800_000_000));
    assert_eq!(result.duration_seconds, Some(3.25));
    assert_eq!(result.usage_duration_seconds, Some(3));
    assert_eq!(result.scope().workspace_id(), "workspace-ab12");
    assert!(!format!("{result:?}").contains("Signature=private"));
    assert!(!format!("{request:?}").contains("reference-secret"));

    let sent = transport.sent();
    assert_eq!(sent.len(), 1, "audio URLs are caller-managed, not fetched");
    assert_eq!(sent[0].method, "POST");
    assert_eq!(
        sent[0].url,
        format!("https://workspace-ab12.cn-beijing.maas.aliyuncs.com{QWEN_AUDIO_GENERATION_PATH}")
    );
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer beijing-key"));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "qwen-audio-3.1-tts-next");
    assert_eq!(body["input"]["text_prompt"], request.text_prompt());
    assert_eq!(body["input"]["format"], "mp3");
    assert_eq!(body["input"]["sample_rate"], 44100);
    assert_eq!(body["input"]["channels"], 1);
    assert_eq!(body["input"]["volume"], 71);
    assert_eq!(body["input"]["enable_cbr"], true);
    assert_eq!(body["input"]["bit_rate"], 192);
    assert_eq!(body["input"]["rate"], 1.25);
    assert_eq!(body["input"]["seed"], 17);
    assert_eq!(body["input"]["enable_aigc_tag"], true);
    assert_eq!(
        body["input"]["references"][0]["audio_data"],
        "data:audio/wav;base64,UklGRi1yZWZlcmVuY2U="
    );
    assert_eq!(
        body["input"]["references"][1]["audio_url"],
        "https://public.example/ref.mp3?token=reference-secret"
    );
}

#[tokio::test]
async fn validates_prompt_references_and_format_controls_before_sending() {
    let transport = MockTransport::new([]);
    let service = QwenAudioGenerationService::new(&transport, scope()).unwrap();
    let options = options();

    let too_long = QwenAudioGenerationRequest::new("x".repeat(3_001));
    assert!(matches!(
        service.synthesize(&too_long, &options).await,
        Err(QwenAudioGenerationError::InvalidInput(_))
    ));

    for prompt in [
        "@voice1 says hello",
        "@voice4 says hello",
        "@voice0 says hello",
    ] {
        assert!(matches!(
            service
                .synthesize(&QwenAudioGenerationRequest::new(prompt), &options)
                .await,
            Err(QwenAudioGenerationError::InvalidInput(_))
        ));
    }

    let too_many_references =
        (0..4).fold(QwenAudioGenerationRequest::new("hello"), |request, _| {
            request.with_reference(
                QwenAudioGenerationReference::from_url("https://public.example/a.wav").unwrap(),
            )
        });
    assert!(matches!(
        service.synthesize(&too_many_references, &options).await,
        Err(QwenAudioGenerationError::InvalidInput(_))
    ));

    for request in [
        QwenAudioGenerationRequest::new("hello").with_volume(101),
        QwenAudioGenerationRequest::new("hello").with_rate(f64::NAN),
        QwenAudioGenerationRequest::new("hello").with_rate(2.1),
        QwenAudioGenerationRequest::new("hello").with_vbr_quality(10),
    ] {
        assert!(matches!(
            service.synthesize(&request, &options).await,
            Err(QwenAudioGenerationError::InvalidInput(_))
        ));
    }

    assert!(QwenAudioGenerationReference::from_url("file:///tmp/voice.wav").is_err());
    assert!(
        QwenAudioGenerationReference::from_url("https://user:pass@example.test/a.wav").is_err()
    );
    assert!(QwenAudioGenerationReference::from_bytes(
        QwenAudioReferenceFormat::OggOpus,
        Bytes::new()
    )
    .is_err());
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn scope_or_missing_credentials_fail_before_http() {
    let transport = MockTransport::new([]);
    let service = QwenAudioGenerationService::new(&transport, scope()).unwrap();
    let request = QwenAudioGenerationRequest::new("hello");
    let wrong_account = RequestOptions {
        account_scope: Some("account-other".into()),
        ..options()
    };
    assert!(matches!(
        service.synthesize(&request, &wrong_account).await,
        Err(QwenAudioGenerationError::ScopeMismatch)
    ));
    assert!(matches!(
        service
            .synthesize(
                &request,
                &RequestOptions {
                    credential: None,
                    ..options()
                }
            )
            .await,
        Err(QwenAudioGenerationError::Llm(
            LlmError::Authentication { .. }
        ))
    ));
    assert!(QwenAudioGenerationScope::new("", "account", "workspace").is_err());
    assert!(QwenAudioGenerationScope::new("profile", "account", "bad.workspace").is_err());
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn zero_total_timeout_is_not_sent_and_not_misclassified_as_unknown() {
    let transport = MockTransport::new([]);
    let service = QwenAudioGenerationService::new(&transport, scope()).unwrap();
    let options = RequestOptions {
        total_timeout: Some(Duration::ZERO),
        ..options()
    };

    let error = service
        .synthesize(&QwenAudioGenerationRequest::new("hello"), &options)
        .await
        .unwrap_err();
    assert!(matches!(error, QwenAudioGenerationError::InvalidInput(_)));
    assert_eq!(error.dispatch(), QwenAudioGenerationDispatch::NotSent);
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn explicit_rejection_unknown_outcome_and_bad_success_are_distinguished() {
    let transport = MockTransport::new([
        reply(
            400,
            json!({"request_id":"bad-1","code":"CLIENT_ERROR","message":"bad prompt"}),
        ),
        reply(
            503,
            json!({"request_id":"maybe-2","message":"temporary failure"}),
        ),
        reply(
            200,
            json!({"request_id":"accepted-3","output":{"finish_reason":"null"}}),
        ),
    ]);
    let service = QwenAudioGenerationService::new(&transport, scope()).unwrap();
    let request = QwenAudioGenerationRequest::new("hello");

    let rejected = service.synthesize(&request, &options()).await.unwrap_err();
    assert_eq!(rejected.dispatch(), QwenAudioGenerationDispatch::Rejected);
    assert!(!format!("{rejected:?}").contains("bad prompt"));
    assert!(matches!(
        rejected,
        QwenAudioGenerationError::Rejected { .. }
    ));

    let unknown = service.synthesize(&request, &options()).await.unwrap_err();
    assert_eq!(unknown.dispatch(), QwenAudioGenerationDispatch::Unknown);
    assert!(matches!(
        unknown,
        QwenAudioGenerationError::OutcomeUnknown { .. }
    ));

    let invalid = service.synthesize(&request, &options()).await.unwrap_err();
    assert_eq!(invalid.dispatch(), QwenAudioGenerationDispatch::Accepted);
    assert!(matches!(
        invalid,
        QwenAudioGenerationError::AcceptedInvalidResponse { .. }
    ));
    assert_eq!(transport.sent().len(), 3);
}

#[tokio::test]
async fn transport_failure_after_post_is_unknown_and_never_retried() {
    let transport = MockTransport::new([Err(LlmError::ProviderInternal {
        message: "connection closed after send".into(),
    })]);
    let service = QwenAudioGenerationService::new(&transport, scope()).unwrap();
    let error = service
        .synthesize(&QwenAudioGenerationRequest::new("hello"), &options())
        .await
        .unwrap_err();

    assert_eq!(error.dispatch(), QwenAudioGenerationDispatch::Unknown);
    assert!(matches!(
        error,
        QwenAudioGenerationError::OutcomeUnknown { .. }
    ));
    assert_eq!(transport.sent().len(), 1);
}
