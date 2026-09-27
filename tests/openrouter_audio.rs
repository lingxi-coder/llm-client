use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    stream::{self, BoxStream},
    StreamExt,
};
use lingxi_llm_client::{
    audio::AudioInput,
    openrouter_audio::{
        OpenRouterAudioDispatch, OpenRouterAudioService, OpenRouterInputAudioFormat,
        OpenRouterSpeechFormat, OpenRouterSpeechInputReferences, OpenRouterSpeechRequest,
        OpenRouterTimestampGranularity, OpenRouterTranscriptionEncoding,
        OpenRouterTranscriptionRequest, OpenRouterTranscriptionResponseFormat,
        OPENROUTER_AUDIO_SPEECH_ENDPOINT, OPENROUTER_AUDIO_TRANSCRIPTIONS_ENDPOINT,
    },
    protocol::{LlmError, Secret},
    HttpRequest, HttpStreamRequest, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::Poll,
};

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct SentRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    content_length: Option<u64>,
}

struct Fixture {
    replies: Mutex<VecDeque<Reply>>,
    sent: Mutex<Vec<SentRequest>>,
    streamed_requests: AtomicUsize,
    stream_observation: Arc<StreamObservation>,
}

#[derive(Default)]
struct StreamObservation {
    transport_entered: std::sync::atomic::AtomicBool,
    multipart_prefix_received: std::sync::atomic::AtomicBool,
}

impl Fixture {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
            streamed_requests: AtomicUsize::new(0),
            stream_observation: Arc::new(StreamObservation::default()),
        }
    }

    fn sent(&self) -> Vec<SentRequest> {
        self.sent.lock().unwrap().drain(..).collect()
    }

    fn streamed_requests(&self) -> usize {
        self.streamed_requests.load(Ordering::SeqCst)
    }

    fn stream_observation(&self) -> Arc<StreamObservation> {
        self.stream_observation.clone()
    }
}

#[async_trait]
impl Transport for Fixture {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(SentRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
            content_length: None,
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected OpenRouter request");
        Ok(StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        self.streamed_requests.fetch_add(1, Ordering::SeqCst);
        self.stream_observation
            .transport_entered
            .store(true, Ordering::SeqCst);
        let HttpStreamRequest {
            method,
            url,
            headers,
            mut body,
            content_length,
            ..
        } = request;
        let mut collected = Vec::new();
        while let Some(chunk) = body.next().await {
            match chunk {
                Ok(chunk) => {
                    if collected.is_empty() {
                        self.stream_observation
                            .multipart_prefix_received
                            .store(true, Ordering::SeqCst);
                    }
                    collected.extend_from_slice(&chunk);
                }
                Err(error) => {
                    self.sent.lock().unwrap().push(SentRequest {
                        method,
                        url,
                        headers,
                        body: collected,
                        content_length: Some(content_length),
                    });
                    return Err(error);
                }
            }
        }
        self.sent.lock().unwrap().push(SentRequest {
            method,
            url,
            headers,
            body: collected,
            content_length: Some(content_length),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected OpenRouter streaming request");
        Ok(StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }
}

struct SendOnlyTransport;

#[async_trait]
impl Transport for SendOnlyTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("multipart STT must use request-body streaming")
    }
}

fn reply(status: u16, headers: Vec<(&str, &str)>, body: impl Into<Vec<u8>>) -> Reply {
    Reply {
        status,
        headers: headers
            .into_iter()
            .map(|(name, value)| (name.into(), value.into()))
            .collect(),
        body: body.into(),
    }
}

fn json_reply(status: u16, value: Value) -> Reply {
    reply(status, vec![], serde_json::to_vec(&value).unwrap())
}

fn input() -> AudioInput {
    AudioInput::from_bytes(
        "voice.wav",
        "audio/wav",
        Bytes::from_static(&[0, 255, 1, 2]),
    )
}

fn input_with_body(
    size_bytes: u64,
    body: BoxStream<'static, Result<Bytes, LlmError>>,
) -> AudioInput {
    AudioInput {
        filename: "voice.wav".into(),
        media_type: "audio/wav".into(),
        size_bytes,
        body,
    }
}

fn polling_input(size_bytes: u64, polls: Arc<AtomicUsize>) -> AudioInput {
    let body = stream::once(async move {
        polls.fetch_add(1, Ordering::SeqCst);
        Ok(Bytes::from_static(b"data"))
    })
    .boxed();
    input_with_body(size_bytes, body)
}

fn options(account: &str) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new(format!("secret-{account}"))),
        account_scope: Some(account.into()),
        ..Default::default()
    }
}

fn header<'a>(request: &'a SentRequest, name: &str) -> &'a str {
    request
        .headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
        .unwrap()
}

fn json_body(request: &SentRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[tokio::test]
async fn multipart_stt_uses_openrouter_route_and_keeps_audio_bytes() {
    let fixture = Fixture::new([json_reply(
        200,
        json!({
            "text":"recognized words",
            "usage":{"seconds":1.2,"total_tokens":3,"cost":0.0004},
        }),
    )]);
    let service = OpenRouterAudioService::new(&fixture);
    let mut request =
        OpenRouterTranscriptionRequest::new("openai/whisper-1", OpenRouterInputAudioFormat::Wav);
    request.language = Some("en".into());
    request.response_format = OpenRouterTranscriptionResponseFormat::VerboseJson;
    request.timestamp_granularities = vec![OpenRouterTimestampGranularity::Word];

    let result = service
        .transcribe(input(), &request, &options("account-a"))
        .await
        .unwrap();

    assert_eq!(result.text, "recognized words");
    assert_eq!(result.usage.as_ref().unwrap().seconds, Some(1.2));
    assert_eq!(result.account_scope.as_deref(), Some("account-a"));
    assert_eq!(fixture.streamed_requests(), 1);
    let sent = fixture.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(sent[0].url, OPENROUTER_AUDIO_TRANSCRIPTIONS_ENDPOINT);
    assert_eq!(header(&sent[0], "authorization"), "Bearer secret-account-a");
    let content_type = header(&sent[0], "content-type");
    let boundary = content_type
        .strip_prefix("multipart/form-data; boundary=")
        .unwrap();
    let body = String::from_utf8_lossy(&sent[0].body);
    assert!(body.contains("name=\"model\"\r\n\r\nopenai/whisper-1\r\n"));
    assert!(body.contains("name=\"language\"\r\n\r\nen\r\n"));
    assert!(body.contains("name=\"response_format\"\r\n\r\nverbose_json\r\n"));
    assert!(body.contains("name=\"timestamp_granularities[]\"\r\n\r\nword\r\n"));
    assert!(body.contains("filename=\"voice.wav\""));
    assert!(body.contains("Content-Type: audio/wav"));
    assert!(sent[0].body.windows(4).any(|part| part == [0, 255, 1, 2]));
    assert!(body.contains(&format!("--{boundary}--\r\n")));
    assert!(!body.contains("account-a"));
    assert_eq!(sent[0].content_length, Some(sent[0].body.len() as u64));
}

#[tokio::test]
async fn multipart_source_is_polled_after_transport_receives_framing_prefix() {
    let fixture = Fixture::new([json_reply(200, json!({"text":"chunked audio"}))]);
    let observation = fixture.stream_observation();
    let mut emitted_audio = false;
    let body = stream::poll_fn(move |_| {
        assert!(observation.transport_entered.load(Ordering::SeqCst));
        assert!(observation.multipart_prefix_received.load(Ordering::SeqCst));
        if emitted_audio {
            Poll::Ready(None)
        } else {
            emitted_audio = true;
            Poll::Ready(Some(Ok(Bytes::from_static(b"chunk"))))
        }
    })
    .boxed();
    let service = OpenRouterAudioService::new(&fixture);
    let request =
        OpenRouterTranscriptionRequest::new("openai/whisper-1", OpenRouterInputAudioFormat::Wav);

    let result = service
        .transcribe(input_with_body(5, body), &request, &options("scope-a"))
        .await
        .unwrap();

    assert_eq!(result.text, "chunked audio");
    assert_eq!(fixture.streamed_requests(), 1);
    let sent = fixture.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].content_length, Some(sent[0].body.len() as u64));
    assert!(sent[0].body.windows(5).any(|chunk| chunk == b"chunk"));
}

#[tokio::test]
async fn json_stt_uses_base64_audio_object_and_scopes_credentials_per_call() {
    let fixture = Fixture::new([
        reply(
            200,
            vec![("X-Generation-Id", "generation-1")],
            br#"{"text":"first","usage":{"input_tokens":2}}"#.to_vec(),
        ),
        reply(
            200,
            vec![("x-generation-id", "generation-2")],
            br#"{"text":"second"}"#.to_vec(),
        ),
    ]);
    let service = OpenRouterAudioService::new(&fixture);
    let mut request = OpenRouterTranscriptionRequest::new(
        "openai/whisper-large-v3",
        OpenRouterInputAudioFormat::Wav,
    );
    request.encoding = OpenRouterTranscriptionEncoding::Base64Json;
    request.language = Some("en".into());
    request.temperature = Some(0.2);
    request.provider = Some(json!({"options":{"groq":{"prompt":"railway names"}}}));

    let first = service
        .transcribe(input(), &request, &options("one"))
        .await
        .unwrap();
    let second = service
        .transcribe(input(), &request, &options("two"))
        .await
        .unwrap();

    assert_eq!(first.generation_id.as_deref(), Some("generation-1"));
    assert_eq!(first.account_scope.as_deref(), Some("one"));
    assert_eq!(second.generation_id.as_deref(), Some("generation-2"));
    assert_eq!(second.account_scope.as_deref(), Some("two"));
    let sent = fixture.sent();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].url, OPENROUTER_AUDIO_TRANSCRIPTIONS_ENDPOINT);
    assert_eq!(header(&sent[0], "content-type"), "application/json");
    assert_eq!(header(&sent[0], "authorization"), "Bearer secret-one");
    assert_eq!(header(&sent[1], "authorization"), "Bearer secret-two");
    let body = json_body(&sent[0]);
    assert_eq!(body["model"], "openai/whisper-large-v3");
    assert_eq!(body["input_audio"]["data"], "AP8BAg==");
    assert_eq!(body["input_audio"]["format"], "wav");
    assert_eq!(body["language"], "en");
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(
        body["provider"]["options"]["groq"]["prompt"],
        "railway names"
    );
    assert!(String::from_utf8_lossy(&sent[0].body)
        .find("\"account_scope\"")
        .is_none());
    assert_eq!(fixture.streamed_requests(), 0);
}

#[tokio::test]
async fn tts_returns_raw_bytes_and_openrouter_generation_metadata() {
    let fixture = Fixture::new([reply(
        200,
        vec![
            ("Content-Type", "audio/mpeg"),
            ("X-Generation-Id", "speech-42"),
        ],
        vec![0, 255, 1, 2],
    )]);
    let service = OpenRouterAudioService::new(&fixture);
    let mut request = OpenRouterSpeechRequest::new("mistralai/voxtral-mini-tts-2603", "Hello");
    request.voice = Some("en_paul_neutral".into());
    request.response_format = OpenRouterSpeechFormat::Mp3;
    request.speed = Some(1.0);
    request.provider = Some(json!({"options":{"microsoft":{"style":"cheerful"}}}));

    let output = service
        .speak(&request, &options("speech-account"))
        .await
        .unwrap();

    assert_eq!(output.bytes.as_ref(), &[0, 255, 1, 2]);
    assert_eq!(output.content_type, "audio/mpeg");
    assert_eq!(output.format, OpenRouterSpeechFormat::Mp3);
    assert_eq!(output.generation_id.as_deref(), Some("speech-42"));
    assert_eq!(output.account_scope.as_deref(), Some("speech-account"));
    let sent = fixture.sent();
    assert_eq!(sent[0].url, OPENROUTER_AUDIO_SPEECH_ENDPOINT);
    assert_eq!(
        header(&sent[0], "authorization"),
        "Bearer secret-speech-account"
    );
    let body = json_body(&sent[0]);
    assert_eq!(body["model"], "mistralai/voxtral-mini-tts-2603");
    assert_eq!(body["input"], "Hello");
    assert_eq!(body["voice"], "en_paul_neutral");
    assert_eq!(body["response_format"], "mp3");
    assert_eq!(body["speed"], 1.0);
    assert_eq!(
        body["provider"]["options"]["microsoft"]["style"],
        "cheerful"
    );
}

#[tokio::test]
async fn tts_serializes_one_typed_base64_reference_and_optional_transcript() {
    let fixture = Fixture::new([reply(
        200,
        vec![("Content-Type", "audio/mpeg")],
        vec![1, 2, 3],
    )]);
    let service = OpenRouterAudioService::new(&fixture);
    let references =
        OpenRouterSpeechInputReferences::new(OpenRouterInputAudioFormat::Wav, "U0VDUkVUX0FVRElP")
            .with_transcript("TRANSCRIPT_SECRET");
    let debug = format!("{references:?}");
    assert!(!debug.contains("U0VDUkVUX0FVRElP"));
    assert!(!debug.contains("TRANSCRIPT_SECRET"));
    assert!(debug.contains("<redacted>"));

    let mut request = OpenRouterSpeechRequest::new("fish-audio/s2.1-pro", "Say this");
    request.input_references = Some(references);
    service
        .speak(&request, &options("voice-account"))
        .await
        .unwrap();

    let sent = fixture.sent();
    let body = json_body(&sent[0]);
    assert_eq!(body["input_references"].as_array().unwrap().len(), 2);
    assert_eq!(body["input_references"][0]["type"], "input_audio");
    assert_eq!(
        body["input_references"][0]["input_audio"]["data"],
        "data:audio/wav;base64,U0VDUkVUX0FVRElP"
    );
    assert_eq!(body["input_references"][1]["type"], "text");
    assert_eq!(body["input_references"][1]["text"], "TRANSCRIPT_SECRET");
}

#[tokio::test]
async fn tts_reference_validation_is_local_but_model_entitlement_is_provider_owned() {
    let no_io = Fixture::new([]);
    let service = OpenRouterAudioService::new(&no_io);
    let mut invalid = OpenRouterSpeechRequest::new("any/model", "Hello");
    invalid.input_references = Some(OpenRouterSpeechInputReferences::new(
        OpenRouterInputAudioFormat::Wav,
        "not base64!",
    ));
    assert!(matches!(
        service.speak(&invalid, &options("a")).await,
        Err(
            lingxi_llm_client::openrouter_audio::OpenRouterAudioError::Llm(
                LlmError::InvalidRequest { .. }
            )
        )
    ));
    assert!(no_io.sent().is_empty());

    let over_limit = "A".repeat(20 * 1024 * 1024 + 4);
    let mut too_large = OpenRouterSpeechRequest::new("any/model", "Hello");
    too_large.input_references = Some(OpenRouterSpeechInputReferences::new(
        OpenRouterInputAudioFormat::Wav,
        over_limit,
    ));
    assert!(matches!(
        service.speak(&too_large, &options("a")).await,
        Err(
            lingxi_llm_client::openrouter_audio::OpenRouterAudioError::Llm(
                LlmError::RequestTooLarge { .. }
            )
        )
    ));
    assert!(no_io.sent().is_empty());

    let provider_owned = Fixture::new([reply(200, vec![("Content-Type", "audio/mpeg")], vec![1])]);
    let service = OpenRouterAudioService::new(&provider_owned);
    let mut request = OpenRouterSpeechRequest::new("unlisted/provider-model", "Hello");
    request.input_references = Some(OpenRouterSpeechInputReferences::new(
        OpenRouterInputAudioFormat::Wav,
        "AQID",
    ));
    service.speak(&request, &options("a")).await.unwrap();
    assert_eq!(provider_owned.sent().len(), 1);
}

#[tokio::test]
async fn provider_rejection_keeps_dispatch_and_error_body() {
    let fixture = Fixture::new([json_reply(
        400,
        json!({"error":{"code":400,"message":"unsupported model"}}),
    )]);
    let service = OpenRouterAudioService::new(&fixture);
    let request = OpenRouterSpeechRequest::new("unknown/model", "Hello");
    let error = service.speak(&request, &options("a")).await.unwrap_err();

    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::Rejected);
    match error {
        lingxi_llm_client::openrouter_audio::OpenRouterAudioError::Provider {
            status,
            message,
            body,
            ..
        } => {
            assert_eq!(status, 400);
            assert_eq!(message, "unsupported model");
            assert_eq!(body.unwrap()["error"]["code"], 400);
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(fixture.sent().len(), 1);
}

#[tokio::test]
async fn tts_rejects_json_success_body_instead_of_returning_it_as_audio() {
    let fixture = Fixture::new([reply(
        200,
        vec![("Content-Type", "application/json")],
        br#"{"error":"unexpected proxy response"}"#.to_vec(),
    )]);
    let service = OpenRouterAudioService::new(&fixture);
    let error = service
        .speak(
            &OpenRouterSpeechRequest::new("model", "Hello"),
            &options("a"),
        )
        .await
        .unwrap_err();

    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::Accepted);
    assert!(matches!(
        error,
        lingxi_llm_client::openrouter_audio::OpenRouterAudioError::InvalidResponse { .. }
    ));
}

#[tokio::test]
async fn multipart_stt_rejects_json_provider_options_before_transport() {
    let fixture = Fixture::new([]);
    let service = OpenRouterAudioService::new(&fixture);
    let mut request =
        OpenRouterTranscriptionRequest::new("openai/whisper-1", OpenRouterInputAudioFormat::Wav);
    request.provider = Some(json!({"options":{"groq":{"prompt":"terms"}}}));

    let error = service
        .transcribe(input(), &request, &options("a"))
        .await
        .unwrap_err();

    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::NotSent);
    assert!(fixture.sent().is_empty());
}

#[tokio::test]
async fn multipart_preflight_and_missing_auth_do_not_poll_audio() {
    let fixture = Fixture::new([]);
    let service = OpenRouterAudioService::new(&fixture);
    let mut request =
        OpenRouterTranscriptionRequest::new("openai/whisper-1", OpenRouterInputAudioFormat::Wav);
    request.provider = Some(json!({"options":{"groq":{"prompt":"terms"}}}));
    let invalid_request_polls = Arc::new(AtomicUsize::new(0));
    let error = service
        .transcribe(
            polling_input(4, invalid_request_polls.clone()),
            &request,
            &options("a"),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::NotSent);
    assert_eq!(invalid_request_polls.load(Ordering::SeqCst), 0);

    request.provider = None;
    let missing_auth_polls = Arc::new(AtomicUsize::new(0));
    let error = service
        .transcribe(
            polling_input(4, missing_auth_polls.clone()),
            &request,
            &RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::NotSent);
    assert_eq!(missing_auth_polls.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.streamed_requests(), 0);
    assert!(fixture.sent().is_empty());

    let no_stream_transport = SendOnlyTransport;
    let service = OpenRouterAudioService::new(&no_stream_transport);
    let no_transport_polls = Arc::new(AtomicUsize::new(0));
    let error = service
        .transcribe(
            polling_input(4, no_transport_polls.clone()),
            &request,
            &options("a"),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::NotSent);
    assert!(matches!(
        error,
        lingxi_llm_client::openrouter_audio::OpenRouterAudioError::Llm(
            LlmError::UnsupportedCapability { .. }
        )
    ));
    assert_eq!(no_transport_polls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn multipart_length_mismatch_is_unknown_and_does_not_emit_closing_boundary() {
    for (declared_size, audio_bytes) in [(5, b"abc".as_slice()), (2, b"abcd".as_slice())] {
        let fixture = Fixture::new([]);
        let service = OpenRouterAudioService::new(&fixture);
        let request = OpenRouterTranscriptionRequest::new(
            "openai/whisper-1",
            OpenRouterInputAudioFormat::Wav,
        );
        let body = stream::iter([Ok(Bytes::copy_from_slice(audio_bytes))]).boxed();

        let error = service
            .transcribe(
                input_with_body(declared_size, body),
                &request,
                &options("scope-a"),
            )
            .await
            .unwrap_err();

        assert_eq!(error.dispatch(), OpenRouterAudioDispatch::Unknown);
        assert_eq!(fixture.streamed_requests(), 1);
        let sent = fixture.sent();
        assert_eq!(sent.len(), 1);
        let content_type = header(&sent[0], "content-type");
        let boundary = content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        let closing = format!("\r\n--{boundary}--\r\n");
        assert!(!sent[0].body.ends_with(closing.as_bytes()));
        assert!(sent[0].content_length.unwrap() > sent[0].body.len() as u64);
    }
}

#[tokio::test]
async fn multipart_source_error_after_dispatch_is_unknown_even_if_invalid_request() {
    let fixture = Fixture::new([]);
    let service = OpenRouterAudioService::new(&fixture);
    let request =
        OpenRouterTranscriptionRequest::new("openai/whisper-1", OpenRouterInputAudioFormat::Wav);
    let body = stream::iter([
        Ok(Bytes::from_static(b"ab")),
        Err(LlmError::InvalidRequest {
            message: "source failed after one chunk".into(),
        }),
    ])
    .boxed();

    let error = service
        .transcribe(input_with_body(4, body), &request, &options("scope-a"))
        .await
        .unwrap_err();

    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::Unknown);
    assert_eq!(fixture.streamed_requests(), 1);
    let sent = fixture.sent();
    assert_eq!(sent.len(), 1);
    assert!(sent[0].body.windows(2).any(|chunk| chunk == b"ab"));
    assert!(!sent[0].body.ends_with(b"--\r\n"));
}

#[tokio::test]
async fn missing_key_fails_before_transport() {
    let fixture = Fixture::new([]);
    let service = OpenRouterAudioService::new(&fixture);
    let error = service
        .speak(
            &OpenRouterSpeechRequest::new("model", "Hello"),
            &RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), OpenRouterAudioDispatch::NotSent);
    assert!(fixture.sent().is_empty());
}
