use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{AuthStrategy, LlmError, ProviderProfile, Region, Secret},
    providers::minimax::audio::{
        MiniMaxAudioDispatch, MiniMaxAudioError, MiniMaxAudioService, MiniMaxSpeechLanguage,
        MiniMaxTimestampLevel, MiniMaxTranscriptFormat, MiniMaxTranscriptionRequest,
        MINIMAX_ASR_CHINA_ENDPOINT, MINIMAX_ASR_INTERNATIONAL_ENDPOINT,
    },
    providers::openai::audio::AudioInput,
    Authenticator, HttpRequest, HttpStreamRequest, LlmClientBuilder, RequestOptions,
    StreamResponse, Transport,
};
use serde_json::json;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct SentRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    declared_length: u64,
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    chunks: Option<Vec<Vec<u8>>>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Reply>>,
    sent: Mutex<Vec<SentRequest>>,
    fail_after_upload: bool,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
            fail_after_upload: false,
        }
    }

    fn failing_after_upload() -> Self {
        Self {
            replies: Mutex::new(VecDeque::new()),
            sent: Mutex::new(Vec::new()),
            fail_after_upload: true,
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("MiniMax audio must use its one-shot multipart streaming upload")
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
            declared_length: content_length,
        });

        if self.fail_after_upload {
            return Err(LlmError::Transport {
                message: "mock connection lost after upload".into(),
            });
        }

        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected MiniMax ASR call");
        Ok(StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: match reply.chunks {
                Some(chunks) => {
                    stream::iter(chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk)))).boxed()
                }
                None => stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
            },
        })
    }
}

fn reply(status: u16, body: impl Into<Vec<u8>>) -> Reply {
    Reply {
        status,
        headers: vec![("x-request-id".into(), "header-trace-123".into())],
        body: body.into(),
        chunks: None,
    }
}

fn chunked_reply(status: u16, chunks: Vec<Vec<u8>>) -> Reply {
    Reply {
        status,
        headers: vec![
            ("x-request-id".into(), "header-trace-123".into()),
            (
                "content-type".into(),
                "text/event-stream; charset=utf-8".into(),
            ),
        ],
        body: Vec::new(),
        chunks: Some(chunks),
    }
}

fn json_reply(value: serde_json::Value) -> Reply {
    reply(200, serde_json::to_vec(&value).unwrap())
}

fn input() -> AudioInput {
    AudioInput::from_bytes(
        "voice.wav",
        "audio/wav",
        Bytes::from_static(&[0, 255, 1, 2]),
    )
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("mini-secret".to_owned())),
        ..Default::default()
    }
}

struct HeaderAuthenticator;

#[async_trait]
impl Authenticator for HeaderAuthenticator {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        _profile: &ProviderProfile,
        credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        request.headers.push((
            "x-custom-auth".into(),
            credential
                .expect("request credential")
                .expose_secret()
                .clone(),
        ));
        Ok(())
    }
}

#[tokio::test]
async fn typed_minimax_asr_uses_registered_authenticator() {
    let transport = Arc::new(MockTransport::new([json_reply(json!({"text":"ok"}))]));
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"minimax", "profile_name":"mini", "protocol":"open_ai_chat",
        "base_url":"https://api.minimax.io/v1", "auth":"api_key",
        "chat_enabled":false, "models":[]
    }))
    .unwrap();
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[profile])
        .with_region(Region::International);
    builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(HeaderAuthenticator));
    let client = builder.build().unwrap();
    let provider = client
        .provider::<lingxi_llm_client::providers::MiniMaxClient>("mini")
        .unwrap();
    let service = provider.audio(MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let result = service
        .transcribe(input(), &MiniMaxTranscriptionRequest::default(), &options())
        .await
        .unwrap();
    assert_eq!(result.text, "ok");
    let sent = transport.sent.lock().unwrap();
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "x-custom-auth" && value == "mini-secret"));
    assert!(!sent[0]
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("authorization")));
}

#[tokio::test]
async fn json_transcription_sends_exact_multipart_to_explicit_regional_endpoint() {
    let transport = MockTransport::new([json_reply(json!({
        "text":"识别结果",
        "duration":2.75,
        "trace_id":"mini-trace",
    }))]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let request = MiniMaxTranscriptionRequest {
        language: Some(MiniMaxSpeechLanguage::Chinese),
        ..Default::default()
    };

    let result = service
        .transcribe(input(), &request, &options())
        .await
        .unwrap();

    assert_eq!(result.text, "识别结果");
    assert_eq!(result.format, MiniMaxTranscriptFormat::Json);
    assert_eq!(result.duration_seconds, Some(2.75));
    assert_eq!(result.request_id.as_deref(), Some("mini-trace"));
    assert_eq!(result.native.as_ref().unwrap()["trace_id"], "mini-trace");

    let sent = transport.sent.lock().unwrap();
    let sent = &sent[0];
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.url, MINIMAX_ASR_INTERNATIONAL_ENDPOINT);
    assert_eq!(sent.declared_length, sent.body.len() as u64);
    assert!(sent.body.windows(4).any(|part| part == [0, 255, 1, 2]));
    let body = String::from_utf8_lossy(&sent.body);
    assert!(body.contains("name=\"model\"\r\n\r\nasr-1.0\r\n"));
    assert!(body.contains("name=\"response_format\"\r\n\r\njson\r\n"));
    assert!(body.contains("name=\"timestamp_level\"\r\n\r\nsentence\r\n"));
    assert!(body.contains("name=\"stream\"\r\n\r\nfalse\r\n"));
    assert!(body.contains("name=\"file\"; filename=\"voice.wav\""));
    assert!(sent
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer mini-secret"));
    assert!(sent
        .headers
        .iter()
        .any(|(name, value)| name == "language" && value == "zh"));
    let content_type = sent
        .headers
        .iter()
        .find(|(name, _)| name == "content-type")
        .unwrap()
        .1
        .clone();
    let boundary = content_type
        .strip_prefix("multipart/form-data; boundary=")
        .unwrap();
    assert!(body.contains(&format!("--{boundary}--\r\n")));
}

#[tokio::test]
async fn streaming_transcription_parses_split_sse_deltas_and_terminal_duration() {
    let first = b"data: {\"index\":0,\"delta\":\"Hello\",\"finish\":false}\r\n\r\n";
    let second = b"data: {\"index\":1,\"delta\":\" world\",\"finish\":true,\"duration\":2.5}\n\n";
    let mut chunks = vec![first[..8].to_vec(), first[8..].to_vec()];
    chunks.extend([second[..17].to_vec(), second[17..].to_vec()]);
    let transport = MockTransport::new([chunked_reply(200, chunks)]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();

    let mut output = service
        .transcribe_stream(input(), &MiniMaxTranscriptionRequest::default(), &options())
        .await
        .unwrap();
    assert_eq!(output.request_id(), Some("header-trace-123"));
    let first = output.next().await.unwrap().unwrap();
    assert_eq!(first.index, 0);
    assert_eq!(first.delta, "Hello");
    assert!(!first.finish);
    assert_eq!(first.duration_seconds, None);
    let final_event = output.next().await.unwrap().unwrap();
    assert_eq!(final_event.index, 1);
    assert_eq!(final_event.delta, " world");
    assert!(final_event.finish);
    assert_eq!(final_event.duration_seconds, Some(2.5));
    assert!(output.next().await.is_none());

    let sent = transport.sent.lock().unwrap();
    let body = String::from_utf8_lossy(&sent[0].body);
    assert!(body.contains("name=\"stream\"\r\n\r\ntrue\r\n"));
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("accept") && value == "text/event-stream"));
}

#[tokio::test]
async fn streaming_requires_json_and_a_valid_terminal_event() {
    let no_io = MockTransport::new([]);
    let service = MiniMaxAudioService::new(&no_io, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let request = MiniMaxTranscriptionRequest {
        format: MiniMaxTranscriptFormat::VerboseJson,
        ..Default::default()
    };
    assert!(matches!(
        service
            .transcribe_stream(input(), &request, &options())
            .await,
        Err(MiniMaxAudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(no_io.sent.lock().unwrap().is_empty());

    let body = b"data: {\"index\":1,\"delta\":\"gap\",\"finish\":true,\"duration\":1.0}\n\n";
    let transport = MockTransport::new([chunked_reply(200, vec![body.to_vec()])]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let mut output = service
        .transcribe_stream(input(), &MiniMaxTranscriptionRequest::default(), &options())
        .await
        .unwrap();
    assert!(matches!(
        output.next().await.unwrap(),
        Err(MiniMaxAudioError::InvalidResponse { .. })
    ));
}

#[tokio::test]
async fn each_call_uses_only_its_request_scoped_credential() {
    let transport = MockTransport::new([
        json_reply(json!({ "text": "first", "duration": 1.0 })),
        json_reply(json!({ "text": "second", "duration": 1.0 })),
    ]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let first = RequestOptions {
        credential: Some(Secret::new("account-one-secret".into())),
        account_scope: Some("account-one".into()),
        ..Default::default()
    };
    let second = RequestOptions {
        credential: Some(Secret::new("account-two-secret".into())),
        account_scope: Some("account-two".into()),
        ..Default::default()
    };
    let request = MiniMaxTranscriptionRequest::default();

    service.transcribe(input(), &request, &first).await.unwrap();
    service
        .transcribe(input(), &request, &second)
        .await
        .unwrap();

    let sent = transport.sent.lock().unwrap();
    let first_auth = sent[0]
        .headers
        .iter()
        .find(|(name, _)| name == "authorization")
        .map(|(_, value)| value.as_str())
        .unwrap();
    let second_auth = sent[1]
        .headers
        .iter()
        .find(|(name, _)| name == "authorization")
        .map(|(_, value)| value.as_str())
        .unwrap();
    assert_eq!(first_auth, "Bearer account-one-secret");
    assert_eq!(second_auth, "Bearer account-two-secret");
    for request in sent.iter() {
        let body = String::from_utf8_lossy(&request.body);
        assert!(!body.contains("account-one"));
        assert!(!body.contains("account-two"));
    }
}

#[tokio::test]
async fn verbose_json_preserves_diarized_segments_and_china_route() {
    let transport = MockTransport::new([json_reply(json!({
        "text":"hello world",
        "duration":1.2,
        "n_speakers":2,
        "segments":[
            {"start":0.0,"end":0.4,"text":"hello","speaker":"speaker_0"},
            {"start":0.4,"end":1.2,"text":" world","speaker":"speaker_1"}
        ]
    }))]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_CHINA_ENDPOINT).unwrap();
    let request = MiniMaxTranscriptionRequest {
        format: MiniMaxTranscriptFormat::VerboseJson,
        timestamp_level: MiniMaxTimestampLevel::Word,
        language: Some(MiniMaxSpeechLanguage::English),
    };

    let result = service
        .transcribe(input(), &request, &options())
        .await
        .unwrap();

    assert_eq!(result.text, "hello world");
    assert_eq!(result.speaker_count, Some(2));
    assert_eq!(result.segments.len(), 2);
    assert_eq!(result.segments[1].speaker.as_deref(), Some("speaker_1"));
    assert_eq!(result.segments[1].start, 0.4);
    assert!(result.native.is_some());
    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent[0].url, MINIMAX_ASR_CHINA_ENDPOINT);
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "language" && value == "en"));
    assert!(String::from_utf8_lossy(&sent[0].body)
        .contains("name=\"response_format\"\r\n\r\nverbose_json\r\n"));
    assert!(
        String::from_utf8_lossy(&sent[0].body).contains("name=\"timestamp_level\"\r\n\r\nword\r\n")
    );
}

#[tokio::test]
async fn subtitle_formats_return_the_complete_srt_or_vtt_document() {
    let transport = MockTransport::new([
        reply(200, b"1\n00:00:00,100 --> 00:00:01,000\nHello.\n".to_vec()),
        reply(
            200,
            b"WEBVTT\n\n00:00:00.100 --> 00:00:01.000\nHello.\n".to_vec(),
        ),
    ]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();

    for (format, expected) in [
        (
            MiniMaxTranscriptFormat::Srt,
            "1\n00:00:00,100 --> 00:00:01,000\nHello.\n",
        ),
        (
            MiniMaxTranscriptFormat::Vtt,
            "WEBVTT\n\n00:00:00.100 --> 00:00:01.000\nHello.\n",
        ),
    ] {
        let result = service
            .transcribe(
                input(),
                &MiniMaxTranscriptionRequest {
                    format,
                    ..Default::default()
                },
                &options(),
            )
            .await
            .unwrap();
        assert_eq!(result.text, expected);
        assert_eq!(result.format, format);
        assert!(result.native.is_none());
        assert_eq!(result.request_id.as_deref(), Some("header-trace-123"));
    }
}

#[tokio::test]
async fn local_errors_do_not_send_and_endpoint_rejects_untrusted_hosts() {
    for endpoint in [
        "http://api.minimax.io/v1/speech_to_text",
        "https://api.minimax.io.attacker.invalid/v1/speech_to_text",
        "https://user@api.minimax.io/v1/speech_to_text",
        "https://@api.minimax.io/v1/speech_to_text",
        "https://api.minimax.io/v1/speech_to_text?key=secret",
        "https://api.minimax.io/v1/other",
    ] {
        assert!(MiniMaxAudioService::new(&MockTransport::new([]), endpoint).is_err());
    }

    let transport = MockTransport::new([]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let mut missing_options = options();
    missing_options.credential = None;
    let missing_key = service
        .transcribe(
            input(),
            &MiniMaxTranscriptionRequest::default(),
            &missing_options,
        )
        .await
        .unwrap_err();
    assert_eq!(missing_key.dispatch(), MiniMaxAudioDispatch::NotSent);

    let invalid_audio = AudioInput::from_bytes("voice.raw", "audio/wav", Bytes::from_static(b"x"));
    let invalid_file = service
        .transcribe(
            invalid_audio,
            &MiniMaxTranscriptionRequest::default(),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(invalid_file.dispatch(), MiniMaxAudioDispatch::NotSent);
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn provider_rejections_and_success_decode_failures_expose_dispatch_state() {
    let transport = MockTransport::new([
        reply(
            429,
            br#"{"base_resp":{"status_code":1008,"status_msg":"rate limited"},"trace_id":"rejected-trace"}"#.to_vec(),
        ),
        reply(200, b"not json".to_vec()),
    ]);
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();

    let rejected = service
        .transcribe(input(), &MiniMaxTranscriptionRequest::default(), &options())
        .await
        .unwrap_err();
    assert_eq!(rejected.dispatch(), MiniMaxAudioDispatch::Rejected);
    assert!(matches!(
        rejected,
        MiniMaxAudioError::Provider {
            status: 429,
            request_id: Some(ref id),
            ..
        } if id == "rejected-trace"
    ));

    let malformed = service
        .transcribe(input(), &MiniMaxTranscriptionRequest::default(), &options())
        .await
        .unwrap_err();
    assert_eq!(malformed.dispatch(), MiniMaxAudioDispatch::Accepted);
    assert!(matches!(
        malformed,
        MiniMaxAudioError::InvalidResponse { .. }
    ));
}

#[tokio::test]
async fn interrupted_request_is_unknown_and_input_length_must_match() {
    let transport = MockTransport::failing_after_upload();
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let error = service
        .transcribe(input(), &MiniMaxTranscriptionRequest::default(), &options())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), MiniMaxAudioDispatch::Unknown);
    assert!(matches!(error, MiniMaxAudioError::OutcomeUnknown { .. }));

    let transport = MockTransport::failing_after_upload();
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT).unwrap();
    let short_input = AudioInput {
        filename: "voice.wav".into(),
        media_type: "audio/wav".into(),
        size_bytes: 5,
        body: stream::iter([Ok(Bytes::from_static(b"1234"))]).boxed(),
    };
    let short = service
        .transcribe(
            short_input,
            &MiniMaxTranscriptionRequest::default(),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(short.dispatch(), MiniMaxAudioDispatch::Unknown);
    assert!(matches!(short, MiniMaxAudioError::OutcomeUnknown { .. }));
}
