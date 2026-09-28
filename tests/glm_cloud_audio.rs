use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt, TryStreamExt};
use lingxi_llm_client::{
    protocol::LlmError,
    protocol::Secret,
    providers::openai::audio::AudioInput,
    providers::zhipu::cloud_audio::{
        GlmCloudAudioDispatch, GlmCloudAudioError, GlmCloudAudioRegion, GlmCloudAudioScope,
        GlmCloudAudioService, GlmCloudSpeechRequest, GlmCloudTranscriptionOutput,
        GlmCloudTranscriptionRequest, GLM_ASR_2512_MODEL, GLM_INTERNATIONAL_API_BASE,
        GLM_MAINLAND_API_BASE, GLM_TTS_MODEL,
    },
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::Value;
use std::sync::Mutex;

enum Reply {
    Response {
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<Bytes>,
    },
    ResponseWithStreamError {
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<Bytes>,
        error: LlmError,
    },
    Error(LlmError),
}

struct RecordedRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Bytes,
    declared_content_length: Option<u64>,
}

struct Mock {
    reply: Mutex<Option<Reply>>,
    requests: Mutex<Vec<RecordedRequest>>,
}

impl Mock {
    fn new(reply: Reply) -> Self {
        Self {
            reply: Mutex::new(Some(reply)),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn respond(&self) -> Result<StreamResponse, LlmError> {
        match self.reply.lock().unwrap().take().expect("unexpected retry") {
            Reply::Error(error) => Err(error),
            Reply::Response {
                status,
                headers,
                chunks,
            } => Ok(StreamResponse {
                status,
                headers,
                body: stream::iter(chunks.into_iter().map(Ok)).boxed(),
            }),
            Reply::ResponseWithStreamError {
                status,
                headers,
                chunks,
                error,
            } => Ok(StreamResponse {
                status,
                headers,
                body: stream::iter(
                    chunks
                        .into_iter()
                        .map(Ok::<Bytes, LlmError>)
                        .chain(std::iter::once(Err(error))),
                )
                .boxed(),
            }),
        }
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(RecordedRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body,
            declared_content_length: None,
        });
        self.respond()
    }

    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        let mut body = Vec::new();
        while let Some(chunk) = request.body.next().await {
            body.extend_from_slice(&chunk?);
        }
        self.requests.lock().unwrap().push(RecordedRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: Bytes::from(body),
            declared_content_length: Some(request.content_length),
        });
        self.respond()
    }
}

fn service(region: GlmCloudAudioRegion, mock: &'static Mock) -> GlmCloudAudioService<'static> {
    let scope = GlmCloudAudioScope::new("glm-cloud", "acct-a", region).unwrap();
    GlmCloudAudioService::new(mock, scope).unwrap()
}

fn audio_input(data: impl Into<Bytes>) -> AudioInput {
    AudioInput::from_bytes("meeting.wav", "audio/wav", data)
}

fn audio_response(chunks: Vec<Bytes>, content_type: &str) -> Reply {
    Reply::Response {
        status: 200,
        headers: vec![
            ("content-type".into(), content_type.into()),
            ("x-request-id".into(), "trace-glm-001".into()),
        ],
        chunks,
    }
}

fn mock(reply: Reply) -> &'static Mock {
    Box::leak(Box::new(Mock::new(reply)))
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> &'a str {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
        .unwrap()
}

#[tokio::test]
async fn mainland_asr_uses_the_official_multipart_route_and_completion_shape() {
    let mock = mock(audio_response(
        vec![Bytes::copy_from_slice(
            r#"{"id":"asr-1","model":"glm-asr-2512","choices":[{"message":{"content":"你好。"}}]}"#
                .as_bytes(),
        )],
        "application/json",
    ));
    let svc = service(GlmCloudAudioRegion::MainlandChina, mock);
    let mut request = GlmCloudTranscriptionRequest::new();
    request.request_id = Some("request-001".into());
    request.user_id = Some("user-0001".into());
    let output = svc
        .transcribe(
            &Secret::new("cn-key".to_owned()),
            audio_input(Bytes::from_static(b"RIFF\x00")),
            &request,
        )
        .await
        .unwrap();

    let GlmCloudTranscriptionOutput::Complete(transcript) = output else {
        panic!("expected complete transcript");
    };
    assert_eq!(transcript.text, "你好。");
    assert_eq!(transcript.model.as_deref(), Some(GLM_ASR_2512_MODEL));
    assert_eq!(transcript.response_id.as_deref(), Some("asr-1"));
    assert_eq!(transcript.request_id.as_deref(), Some("trace-glm-001"));
    assert_eq!(transcript.scope.account_scope(), "acct-a");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.url,
        format!("{GLM_MAINLAND_API_BASE}/audio/transcriptions")
    );
    assert_eq!(header(&request.headers, "authorization"), "Bearer cn-key");
    assert_eq!(
        request.declared_content_length,
        Some(request.body.len() as u64)
    );
    let body = String::from_utf8_lossy(&request.body);
    assert!(body.contains("name=\"file\"; filename=\"meeting.wav\""));
    assert!(body.contains(&format!("name=\"model\"\r\n\r\n{GLM_ASR_2512_MODEL}")));
    assert!(body.contains("name=\"stream\"\r\n\r\nfalse"));
    assert!(body.contains("name=\"request_id\"\r\n\r\nrequest-001"));
    assert!(body.contains("name=\"user_id\"\r\n\r\nuser-0001"));
}

#[tokio::test]
async fn international_asr_uses_z_ai_route_and_text_response() {
    let mock = mock(audio_response(
        vec![Bytes::from_static(
            br#"{"id":"asr-intl-1","model":"glm-asr-2512","request_id":"provider-id","text":"Hello there."}"#,
        )],
        "application/json",
    ));
    let svc = service(GlmCloudAudioRegion::International, mock);
    let output = svc
        .transcribe(
            &Secret::new("intl-key".to_owned()),
            audio_input(Bytes::from_static(b"RIFF\x00")),
            &GlmCloudTranscriptionRequest::new(),
        )
        .await
        .unwrap();
    let GlmCloudTranscriptionOutput::Complete(transcript) = output else {
        panic!("expected complete transcript");
    };
    assert_eq!(transcript.text, "Hello there.");
    assert_eq!(transcript.request_id.as_deref(), Some("provider-id"));
    assert_eq!(
        mock.requests.lock().unwrap()[0].url,
        format!("{GLM_INTERNATIONAL_API_BASE}/audio/transcriptions")
    );
}

#[tokio::test]
async fn asr_stream_is_returned_as_opaque_bytes() {
    let mock = mock(audio_response(
        vec![
            Bytes::from_static(b"data: {\"text\":\"Hello"),
            Bytes::from_static(b"\"}\n\n"),
        ],
        "text/event-stream",
    ));
    let svc = service(GlmCloudAudioRegion::International, mock);
    let mut request = GlmCloudTranscriptionRequest::new();
    request.stream = true;
    let output = svc
        .transcribe(
            &Secret::new("intl-key".to_owned()),
            audio_input(Bytes::from_static(b"RIFF\x00")),
            &request,
        )
        .await
        .unwrap();
    let GlmCloudTranscriptionOutput::Streaming(stream) = output else {
        panic!("expected streamed response");
    };
    assert_eq!(stream.content_type(), Some("text/event-stream"));
    assert_eq!(stream.request_id(), Some("trace-glm-001"));
    let chunks = stream.into_body().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(
        chunks,
        vec![
            Bytes::from_static(b"data: {\"text\":\"Hello"),
            Bytes::from_static(b"\"}\n\n")
        ]
    );
}

#[tokio::test]
async fn mainland_tts_streams_binary_response_without_decoding_or_buffering() {
    let mock = mock(audio_response(
        vec![
            Bytes::from_static(b"\x00\xffRI"),
            Bytes::from_static(b"FF\x01\x80"),
        ],
        "audio/wav",
    ));
    let svc = service(GlmCloudAudioRegion::MainlandChina, mock);
    let mut request = GlmCloudSpeechRequest::new("你好", "base64");
    request.voice = Some("tongtong".into());
    request.response_format = Some("wav".into());
    request.stream = true;
    let output = svc
        .synthesize(&Secret::new("cn-key".to_owned()), &request)
        .await
        .unwrap();
    assert_eq!(output.content_type(), Some("audio/wav"));
    assert_eq!(output.scope().region(), GlmCloudAudioRegion::MainlandChina);
    let chunks = output.into_body().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(
        chunks,
        vec![
            Bytes::from_static(b"\x00\xffRI"),
            Bytes::from_static(b"FF\x01\x80")
        ]
    );

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert_eq!(sent.url, format!("{GLM_MAINLAND_API_BASE}/audio/speech"));
    assert_eq!(header(&sent.headers, "content-type"), "application/json");
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["model"], GLM_TTS_MODEL);
    assert_eq!(body["input"], "你好");
    assert_eq!(body["voice"], "tongtong");
    assert_eq!(body["response_format"], "wav");
    assert_eq!(body["encode_format"], "base64");
    assert!(body["stream"].as_bool().unwrap());
}

#[tokio::test]
async fn interrupted_tts_body_remains_an_unknown_billable_outcome() {
    let mock = mock(Reply::ResponseWithStreamError {
        status: 200,
        headers: vec![("content-type".into(), "audio/wav".into())],
        chunks: vec![Bytes::from_static(b"\x00\xff")],
        error: LlmError::StreamInterrupted {
            message: "audio response stopped".into(),
        },
    });
    let svc = service(GlmCloudAudioRegion::MainlandChina, mock);
    let output = svc
        .synthesize(
            &Secret::new("cn-key".to_owned()),
            &GlmCloudSpeechRequest::new("Hello", "base64"),
        )
        .await
        .unwrap();
    let error = output
        .into_body()
        .try_collect::<Vec<_>>()
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GlmCloudAudioDispatch::Unknown);
    assert!(matches!(error, GlmCloudAudioError::OutcomeUnknown { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn international_tts_is_rejected_before_transport() {
    let mock = mock(audio_response(vec![], "audio/wav"));
    let svc = service(GlmCloudAudioRegion::International, mock);
    let error = svc
        .synthesize(
            &Secret::new("intl-key".to_owned()),
            &GlmCloudSpeechRequest::new("Hello", "base64"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        &error,
        GlmCloudAudioError::UnsupportedRegion {
            operation:
                lingxi_llm_client::providers::zhipu::cloud_audio::GlmCloudAudioOperation::Speech,
            region: GlmCloudAudioRegion::International,
        }
    ));
    assert_eq!(error.dispatch(), GlmCloudAudioDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_transport_failure_after_asr_submission_is_unknown_and_never_retried() {
    let mock = mock(Reply::Error(LlmError::Transport {
        message: "connection closed".into(),
    }));
    let svc = service(GlmCloudAudioRegion::MainlandChina, mock);
    let error = svc
        .transcribe(
            &Secret::new("cn-key".to_owned()),
            audio_input(Bytes::from_static(b"RIFF\x00")),
            &GlmCloudTranscriptionRequest::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GlmCloudAudioDispatch::Unknown);
    assert!(matches!(error, GlmCloudAudioError::OutcomeUnknown { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn provider_client_errors_are_rejected_with_request_id() {
    let mock = mock(Reply::Response {
        status: 400,
        headers: vec![("content-type".into(), "application/json".into())],
        chunks: vec![Bytes::from_static(
            br#"{"error":{"message":"bad audio"},"request_id":"provider-400"}"#,
        )],
    });
    let svc = service(GlmCloudAudioRegion::International, mock);
    let error = svc
        .transcribe(
            &Secret::new("intl-key".to_owned()),
            audio_input(Bytes::from_static(b"ID3")),
            &GlmCloudTranscriptionRequest::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GlmCloudAudioDispatch::Rejected);
    assert!(matches!(
        &error,
        GlmCloudAudioError::Provider {
            status: 400,
            request_id: Some(id),
            message,
            ..
        } if id == "provider-400" && message == "bad audio"
    ));
}

#[tokio::test]
async fn malformed_or_oversized_inputs_fail_before_request() {
    let mock = mock(audio_response(vec![], "application/json"));
    let svc = service(GlmCloudAudioRegion::MainlandChina, mock);
    let bad_audio =
        AudioInput::from_bytes("recording.ogg", "audio/ogg", Bytes::from_static(b"OggS"));
    let error = svc
        .transcribe(
            &Secret::new("cn-key".to_owned()),
            bad_audio,
            &GlmCloudTranscriptionRequest::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GlmCloudAudioDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());

    let oversized_audio = AudioInput {
        filename: "large.wav".into(),
        media_type: "audio/wav".into(),
        size_bytes: 25_000_001,
        body: stream::empty::<Result<Bytes, LlmError>>().boxed(),
    };
    let error = svc
        .transcribe(
            &Secret::new("cn-key".to_owned()),
            oversized_audio,
            &GlmCloudTranscriptionRequest::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GlmCloudAudioDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());

    let bad_speech = GlmCloudSpeechRequest::new("\0", "base64");
    let error = svc
        .synthesize(&Secret::new("cn-key".to_owned()), &bad_speech)
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GlmCloudAudioDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());
}
