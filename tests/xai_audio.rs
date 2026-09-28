use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::openai::audio::AudioInput,
    providers::xai::audio::{
        XaiAudioConfig, XaiAudioCredentials, XaiAudioError, XaiAudioService, XaiSpeechOutput,
        XaiSpeechRequest, XaiTranscriptionRequest,
    },
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex, time::Duration};

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    chunks: Vec<Vec<u8>>,
}

struct SentRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    declared_length: Option<u64>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Reply>>,
    sent: Mutex<Vec<SentRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
        }
    }

    fn reply(&self) -> StreamResponse {
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("reply queued");
        StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: futures::stream::iter(
                reply.chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk))),
            )
            .boxed(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(SentRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
            declared_length: None,
        });
        Ok(self.reply())
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let HttpStreamRequest {
            method,
            url,
            headers,
            mut body,
            content_length,
            ..
        } = request;
        let mut bytes = Vec::new();
        while let Some(chunk) = body.next().await {
            bytes.extend_from_slice(&chunk?);
        }
        self.sent.lock().unwrap().push(SentRequest {
            method,
            url,
            headers,
            body: bytes,
            declared_length: Some(content_length),
        });
        Ok(self.reply())
    }
}

fn reply(status: u16, headers: Vec<(String, String)>, chunks: Vec<Vec<u8>>) -> Reply {
    Reply {
        status,
        headers,
        chunks,
    }
}

fn credentials() -> XaiAudioCredentials {
    XaiAudioCredentials::new(Secret::new("xai-test-key".to_owned()))
}

fn service(transport: &MockTransport) -> XaiAudioService<'_> {
    XaiAudioService::new(
        transport,
        XaiAudioConfig::new("work", "team/account-7")
            .with_api_base_url("http://127.0.0.1:8391/v1")
            .with_request_timeout(Duration::from_secs(2)),
    )
    .unwrap()
}

#[tokio::test]
async fn transcription_streams_multipart_with_options_before_the_file_and_scopes_result() {
    let transport = MockTransport::new([reply(
        200,
        vec![("x-request-id".into(), "xai-stt-1".into())],
        vec![
            br#"{"text":"hello world","language":"en","duration":1.25,"words":[{"text":"hello","start":0.0,"end":0.5,"speaker":2}],"channels":[{"index":0,"text":"hello world","words":[]},{"index":1,"text":"quiet","words":[]}] }"#.to_vec(),
        ],
    )]);
    let service = service(&transport);
    let input = AudioInput::from_bytes(
        "meeting.mkv",
        "video/x-matroska",
        Bytes::from_static(b"audio-data"),
    );
    let request = XaiTranscriptionRequest {
        language: Some("en".into()),
        format: true,
        multichannel: true,
        diarize: true,
        keyterms: vec!["LingXi".into(), "xAI voice".into()],
        ..XaiTranscriptionRequest::default()
    };
    let transcript = service
        .transcribe(input, &request, &credentials())
        .await
        .unwrap();

    assert_eq!(transcript.text, "hello world");
    assert_eq!(transcript.language.as_deref(), Some("en"));
    assert_eq!(transcript.duration_seconds, Some(1.25));
    assert_eq!(transcript.words[0].speaker, Some(2));
    assert_eq!(transcript.channels.len(), 2);
    assert_eq!(transcript.scope.account_scope, "team/account-7");
    assert_eq!(transcript.request_id.as_deref(), Some("xai-stt-1"));

    let sent = transport.sent.lock().unwrap();
    let call = &sent[0];
    assert_eq!(call.method, "POST");
    assert_eq!(call.url, "http://127.0.0.1:8391/v1/stt");
    assert_eq!(call.declared_length, Some(call.body.len() as u64));
    assert!(call
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer xai-test-key"));
    let text = String::from_utf8_lossy(&call.body);
    let format_index = text.find("name=\"format\"").unwrap();
    let keyterm_index = text.find("name=\"keyterm\"").unwrap();
    let file_index = text.find("name=\"file\"").unwrap();
    assert!(format_index < keyterm_index && keyterm_index < file_index);
    assert!(text.contains("name=\"keyterm\"\r\n\r\nLingXi"));
    assert!(text.contains("filename=\"meeting.mkv\""));
    assert!(text.ends_with("\r\n"));
}

#[tokio::test]
async fn ordinary_tts_returns_the_provider_raw_audio_stream() {
    let transport = MockTransport::new([reply(
        200,
        vec![
            ("content-type".into(), "audio/mpeg".into()),
            ("request-id".into(), "xai-tts-2".into()),
        ],
        vec![b"ID3".to_vec(), b"-audio".to_vec()],
    )]);
    let service = service(&transport);
    let request = XaiSpeechRequest::new("Hello", "eve", "en");
    let output = service.synthesize(&request, &credentials()).await.unwrap();
    let XaiSpeechOutput::Audio(mut audio) = output else {
        panic!("ordinary TTS is raw audio");
    };
    assert_eq!(audio.content_type, "audio/mpeg");
    assert_eq!(audio.request_id.as_deref(), Some("xai-tts-2"));
    assert_eq!(audio.scope.account_scope, "team/account-7");
    assert_eq!(
        audio.next_chunk().await.unwrap().unwrap(),
        Bytes::from_static(b"ID3")
    );
    assert_eq!(
        audio.next_chunk().await.unwrap().unwrap(),
        Bytes::from_static(b"-audio")
    );
    assert!(audio.next_chunk().await.unwrap().is_none());

    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent[0].url, "http://127.0.0.1:8391/v1/tts");
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["voice_id"], "eve");
    assert_eq!(body["output_format"]["codec"], "mp3");
    assert_eq!(body["output_format"]["sample_rate"], 24_000);
    assert!(body.get("speed").is_none());
    assert!(body.get("optimize_streaming_latency").is_none());
    assert!(body.get("text_normalization").is_none());
    assert!(body.get("replace").is_none());
}

#[tokio::test]
async fn rest_tts_encodes_the_published_speed_latency_normalization_and_replace_fields() {
    let transport = MockTransport::new([reply(
        200,
        vec![
            ("content-type".into(), "audio/mpeg".into()),
            ("request-id".into(), "xai-tts-options".into()),
        ],
        vec![b"ID3".to_vec()],
    )]);
    let service = service(&transport);
    let mut request = XaiSpeechRequest::new("Welcome to Acme Mobile.", "eve", "en");
    request.speed = Some(1.25);
    request.optimize_streaming_latency = Some(2);
    request.text_normalization = true;
    request.replace = [("Acme Mobile".into(), "Acme Mobull".into())]
        .into_iter()
        .collect();

    let _output = service.synthesize(&request, &credentials()).await.unwrap();
    let sent = transport.sent.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["speed"], 1.25);
    assert_eq!(body["optimize_streaming_latency"], 2);
    assert_eq!(body["text_normalization"], true);
    assert_eq!(body["replace"], json!({"Acme Mobile":"Acme Mobull"}));
}

#[tokio::test]
async fn documented_tts_scalar_and_replace_limits_fail_before_http() {
    let transport = MockTransport::new([]);
    let service = service(&transport);

    let mut invalid = Vec::new();
    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.speed = Some(0.69);
    invalid.push(request);

    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.speed = Some(f64::NAN);
    invalid.push(request);

    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.optimize_streaming_latency = Some(3);
    invalid.push(request);

    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.replace = (0..201)
        .map(|index| (format!("term{index}"), "spoken".to_owned()))
        .collect();
    invalid.push(request);

    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.replace = [("k".repeat(101), "spoken".to_owned())]
        .into_iter()
        .collect();
    invalid.push(request);

    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.replace = [("C++".to_owned(), "C plus plus".to_owned())]
        .into_iter()
        .collect();
    invalid.push(request);

    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.replace = [("term".to_owned(), "v".repeat(129))].into_iter().collect();
    invalid.push(request);

    let mut request = XaiSpeechRequest::new("Hello", "eve", "en");
    request.replace = [
        ("Acme Mobile".to_owned(), "Acme Mobull".to_owned()),
        ("ACME   MOBILE".to_owned(), "Acme Mobile".to_owned()),
    ]
    .into_iter()
    .collect();
    invalid.push(request);

    for request in invalid {
        assert!(matches!(
            service.synthesize(&request, &credentials()).await,
            Err(XaiAudioError::InvalidRequest(_))
        ));
    }
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn timestamped_tts_decodes_the_json_base64_audio_envelope() {
    let transport = MockTransport::new([reply(
        200,
        vec![("content-type".into(), "application/json".into())],
        vec![
            br#"{"audio":"AQID","content_type":"audio/mpeg","duration":0.92,"audio_timestamps":{"graph_chars":["H","i"],"graph_times":[[0.0,0.3],[0.3,0.92]]}}"#.to_vec(),
        ],
    )]);
    let service = service(&transport);
    let mut request = XaiSpeechRequest::new("Hi", "eve", "en");
    request.with_timestamps = true;
    let output = service.synthesize(&request, &credentials()).await.unwrap();
    let XaiSpeechOutput::Timestamped(audio) = output else {
        panic!("timestamped TTS is a decoded JSON envelope");
    };
    assert_eq!(audio.audio, Bytes::from_static(&[1, 2, 3]));
    assert_eq!(audio.content_type, "audio/mpeg");
    assert_eq!(audio.duration_seconds, 0.92);
    let timestamps = audio.audio_timestamps.unwrap();
    assert_eq!(timestamps.graph_chars, vec!["H", "i"]);
    assert_eq!(timestamps.graph_times, vec![[0.0, 0.3], [0.3, 0.92]]);
    let body: Value = serde_json::from_slice(&transport.sent.lock().unwrap()[0].body).unwrap();
    assert_eq!(body["with_timestamps"], true);
}

#[tokio::test]
async fn provider_errors_keep_status_request_id_and_json_body() {
    let transport = MockTransport::new([reply(
        429,
        vec![("x-request-id".into(), "xai-limit-3".into())],
        vec![br#"{"error":{"message":"slow down"}}"#.to_vec()],
    )]);
    let error = service(&transport)
        .synthesize(&XaiSpeechRequest::new("Hello", "eve", "en"), &credentials())
        .await
        .unwrap_err();
    match error {
        XaiAudioError::Provider {
            status,
            request_id,
            body,
            ..
        } => {
            assert_eq!(status, 429);
            assert_eq!(request_id.as_deref(), Some("xai-limit-3"));
            assert_eq!(body["error"]["message"], "slow down");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn invalid_requests_fail_before_any_audio_is_sent() {
    let transport = MockTransport::new([]);
    let service = service(&transport);
    let stt = XaiTranscriptionRequest {
        format: true,
        ..XaiTranscriptionRequest::default()
    };
    let error = service
        .transcribe(
            AudioInput::from_bytes("voice.wav", "audio/wav", Bytes::from_static(b"x")),
            &stt,
            &credentials(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, XaiAudioError::InvalidRequest(_)));

    let mut tts = XaiSpeechRequest::new("Hello", "eve", "en");
    tts.output_format.sample_rate = 11_025;
    let error = service.synthesize(&tts, &credentials()).await.unwrap_err();
    assert!(matches!(error, XaiAudioError::InvalidRequest(_)));
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[test]
fn service_requires_nonempty_account_scope_and_secure_nonlocal_endpoint() {
    let transport = MockTransport::new([]);
    let missing_scope = XaiAudioService::new(&transport, XaiAudioConfig::new("work", "  "));
    assert!(matches!(
        missing_scope,
        Err(XaiAudioError::InvalidRequest(_))
    ));
    let insecure = XaiAudioService::new(
        &transport,
        XaiAudioConfig::new("work", "account").with_api_base_url("http://example.test/v1"),
    );
    assert!(matches!(insecure, Err(XaiAudioError::InvalidRequest(_))));
}
