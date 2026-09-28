use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderId, Secret},
    providers::minimax::tts::{
        MiniMaxTtsAudioFormat, MiniMaxTtsConfig, MiniMaxTtsCredentials, MiniMaxTtsDispatch,
        MiniMaxTtsError, MiniMaxTtsModel, MiniMaxTtsOutput, MiniMaxTtsOutputFormat,
        MiniMaxTtsRegion, MiniMaxTtsRequest, MiniMaxTtsService, MiniMaxTtsStreamError,
        MiniMaxTtsStreamEvent, MiniMaxTtsSubtitleType,
    },
    providers::minimax::voices::{
        MiniMaxVoiceKind, MiniMaxVoiceRef, MiniMaxVoicesRegion, MiniMaxVoicesScope,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex, time::Duration};

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    chunks: Option<Vec<Bytes>>,
    stream_error_after_chunks: Option<String>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("queued reply");
        Ok(StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: if let Some(chunks) = reply.chunks {
                let mut events = chunks.into_iter().map(Ok).collect::<Vec<_>>();
                if let Some(message) = reply.stream_error_after_chunks {
                    events.push(Err(LlmError::StreamInterrupted { message }));
                }
                futures::stream::iter(events).boxed()
            } else {
                futures::stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed()
            },
        })
    }
}

fn reply(status: u16, body: Value) -> Reply {
    Reply {
        status,
        headers: vec![("x-request-id".into(), "minimax-header-id".into())],
        body: serde_json::to_vec(&body).unwrap(),
        chunks: None,
        stream_error_after_chunks: None,
    }
}

fn sse_reply(chunks: Vec<Bytes>) -> Reply {
    Reply {
        status: 200,
        headers: vec![
            (
                "content-type".into(),
                "text/event-stream; charset=utf-8".into(),
            ),
            ("x-request-id".into(), "minimax-stream-header-id".into()),
        ],
        body: Vec::new(),
        chunks: Some(chunks),
        stream_error_after_chunks: None,
    }
}

fn make_service(transport: &MockTransport) -> MiniMaxTtsService<'_> {
    MiniMaxTtsService::new(
        transport,
        MiniMaxTtsConfig::new(
            "cn-voice",
            "team/account-17",
            MiniMaxTtsRegion::ChinaMainland,
        )
        .with_endpoint("http://127.0.0.1:8418/v1/t2a_v2")
        .with_request_timeout(Duration::from_secs(3)),
    )
    .unwrap()
}

fn credentials() -> MiniMaxTtsCredentials {
    MiniMaxTtsCredentials::new(Secret::new("minimax-test-key".to_owned()))
}

fn voice_reference(
    provider_id: &str,
    profile_name: &str,
    endpoint_root: &str,
    account_scope: &str,
    region: MiniMaxVoicesRegion,
) -> MiniMaxVoiceRef {
    MiniMaxVoiceRef {
        scope: MiniMaxVoicesScope {
            provider_id: ProviderId::new(provider_id),
            profile_name: profile_name.to_owned(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(endpoint_root),
            account_scope: account_scope.to_owned(),
            region,
        },
        kind: MiniMaxVoiceKind::Cloned,
        voice_id: "created-voice-17".into(),
    }
}

fn success(audio: &str) -> Value {
    json!({
        "data": {"audio": audio, "status": 2},
        "trace_id": "minimax-trace-1",
        "extra_info": {
            "audio_length": 1250,
            "audio_sample_rate": 32000,
            "audio_size": audio.len() / 2,
            "bitrate": 128000,
            "word_count": 3,
            "usage_characters": 10,
            "audio_format": "mp3",
            "audio_channel": 1
        },
        "base_resp": {"status_code": 0, "status_msg": "success"}
    })
}

#[tokio::test]
async fn hex_output_is_decoded_and_keeps_trace_metadata_and_scope() {
    let transport = MockTransport::new([reply(200, success("494433000102ff"))]);
    let service = make_service(&transport);
    let result = service
        .synthesize(
            &MiniMaxTtsRequest::new("Hello from MiniMax", "voice-123"),
            &credentials(),
        )
        .await
        .unwrap();
    let MiniMaxTtsOutput::Audio(audio) = result else {
        panic!("hex response must produce audio bytes");
    };
    assert_eq!(audio.bytes, Bytes::from_static(b"ID3\0\x01\x02\xff"));
    assert_eq!(audio.scope.region, MiniMaxTtsRegion::ChinaMainland);
    assert_eq!(audio.scope.account_scope, "team/account-17");
    assert_eq!(audio.trace_id.as_deref(), Some("minimax-trace-1"));
    assert_eq!(audio.request_id.as_deref(), Some("minimax-trace-1"));
    assert_eq!(audio.extra_info.unwrap().audio_size, Some(7));

    let requests = transport.requests.lock().unwrap();
    let request = &requests[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.url, "http://127.0.0.1:8418/v1/t2a_v2");
    assert!(request
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer minimax-test-key"));
    assert!(request
        .headers
        .iter()
        .any(|(name, value)| name == "content-type" && value == "application/json"));
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["model"], "speech-2.8-hd");
    assert_eq!(body["stream"], false);
    assert_eq!(body["text"], "Hello from MiniMax");
    assert_eq!(body["voice_setting"]["voice_id"], "voice-123");
    assert_eq!(body["audio_setting"]["format"], "mp3");
    assert_eq!(body["output_format"], "hex");
    assert_eq!(body["subtitle_enable"], false);
    assert_eq!(body["subtitle_type"], "sentence");
}

#[tokio::test]
async fn http_tts_sends_documented_subtitle_controls() {
    let transport = MockTransport::new([reply(200, success("494433"))]);
    let service = make_service(&transport);
    let mut request = MiniMaxTtsRequest::new("Speak with word subtitles", "voice-123");
    request.subtitle_enable = true;
    request.subtitle_type = MiniMaxTtsSubtitleType::Word;

    service.synthesize(&request, &credentials()).await.unwrap();

    let requests = transport.requests.lock().unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["stream"], false);
    assert_eq!(body["subtitle_enable"], true);
    assert_eq!(body["subtitle_type"], "word");
}

#[tokio::test]
async fn http_tts_stream_decodes_hex_events_preserves_native_data_and_stops_at_status_two() {
    let wire = concat!(
        "data: {\"data\":{\"audio\":\"494433\",\"status\":1},\"trace_id\":\"stream-trace\",\"subtitle_file\":\"https://cdn.example/subs.json\",\"base_resp\":{\"status_code\":0}}\n\n",
        "data: {\"data\":{\"audio\":\"0001\",\"status\":1},\"base_resp\":{\"status_code\":0}}\n\n",
        "data: {\"data\":{\"audio\":\"02\",\"status\":2},\"base_resp\":{\"status_code\":0}}\n\n",
        "data: [DONE]\n\n"
    );
    let chunks = wire
        .as_bytes()
        .chunks(7)
        .map(Bytes::copy_from_slice)
        .collect();
    let transport = MockTransport::new([sse_reply(chunks)]);
    let service = make_service(&transport);
    let mut request = MiniMaxTtsRequest::new("Stream speech", "voice-123");
    request.subtitle_enable = true;
    request.subtitle_type = MiniMaxTtsSubtitleType::WordStreaming;
    let mut stream = service
        .synthesize_stream(&request, &credentials())
        .await
        .unwrap();
    assert_eq!(stream.scope.account_scope, "team/account-17");
    assert_eq!(
        stream.request_id.as_deref(),
        Some("minimax-stream-header-id")
    );

    let Some(MiniMaxTtsStreamEvent::Data(first)) = stream.next_event().await.unwrap() else {
        panic!("expected first audio event");
    };
    assert_eq!(first.status, Some(1));
    assert_eq!(first.audio, Some(Bytes::from_static(b"ID3")));
    assert_eq!(
        first.native["subtitle_file"],
        "https://cdn.example/subs.json"
    );
    assert_eq!(stream.request_id.as_deref(), Some("stream-trace"));

    let Some(MiniMaxTtsStreamEvent::Data(second)) = stream.next_event().await.unwrap() else {
        panic!("expected second audio event");
    };
    assert_eq!(second.audio, Some(Bytes::from_static(b"\0\x01")));
    let Some(MiniMaxTtsStreamEvent::Data(terminal)) = stream.next_event().await.unwrap() else {
        panic!("expected status-2 event");
    };
    assert_eq!(terminal.status, Some(2));
    assert_eq!(terminal.audio, Some(Bytes::from_static(b"\x02")));
    assert!(stream.terminal_event_received());
    assert!(stream.next_event().await.unwrap().is_none());

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let wire_request = &requests[0];
    assert!(wire_request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("accept") && value == "text/event-stream"
    }));
    let body: Value = serde_json::from_slice(&wire_request.body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["output_format"], "hex");
    assert_eq!(body["audio_setting"]["format"], "mp3");
    assert_eq!(body["subtitle_enable"], true);
    assert_eq!(body["subtitle_type"], "word_streaming");
    assert!(body.get("stream_options").is_none());
}

#[tokio::test]
async fn http_tts_stream_exposes_done_sentinel_and_clean_eof_after_audio() {
    let transport = MockTransport::new([sse_reply(vec![Bytes::from_static(
        b"data: {\"data\":{\"audio\":\"6162\",\"status\":1}}\n\ndata: [DONE]\n\n",
    )])]);
    let service = make_service(&transport);
    let mut stream = service
        .synthesize_stream(
            &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
            &credentials(),
        )
        .await
        .unwrap();
    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(MiniMaxTtsStreamEvent::Data(_))
    ));
    assert_eq!(
        stream.next_event().await.unwrap(),
        Some(MiniMaxTtsStreamEvent::Done)
    );
    assert!(stream.terminal_event_received());
    assert!(stream.next_event().await.unwrap().is_none());

    let transport = MockTransport::new([sse_reply(vec![Bytes::from_static(
        b"data: {\"data\":{\"audio\":\"6162\",\"status\":1}}\n\n",
    )])]);
    let service = make_service(&transport);
    let mut stream = service
        .synthesize_stream(
            &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
            &credentials(),
        )
        .await
        .unwrap();
    assert!(stream.next_event().await.unwrap().is_some());
    assert!(stream.next_event().await.unwrap().is_none());
    assert!(!stream.terminal_event_received());
}

#[tokio::test]
async fn http_tts_stream_preserves_provider_and_transport_failures_with_partial_audio() {
    let wire = concat!(
        "data: {\"data\":{\"audio\":\"6162\",\"status\":1},\"trace_id\":\"partial-trace\"}\n\n",
        "data: {\"base_resp\":{\"status_code\":2056,\"status_msg\":\"quota unavailable\"}}\n\n"
    );
    let transport = MockTransport::new([sse_reply(vec![Bytes::copy_from_slice(wire.as_bytes())])]);
    let service = make_service(&transport);
    let mut stream = service
        .synthesize_stream(
            &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
            &credentials(),
        )
        .await
        .unwrap();
    assert!(stream.next_event().await.unwrap().is_some());
    match stream.next_event().await.unwrap_err() {
        MiniMaxTtsStreamError::Provider {
            bytes_delivered,
            request_id,
            code,
            dispatch,
            ..
        } => {
            assert_eq!(bytes_delivered, 2);
            assert_eq!(request_id.as_deref(), Some("partial-trace"));
            assert_eq!(code, Some(2056));
            assert_eq!(dispatch, MiniMaxTtsDispatch::Accepted);
        }
        other => panic!("unexpected stream error: {other}"),
    }
    assert!(!stream.terminal_event_received());

    let mut interrupted = sse_reply(vec![Bytes::from_static(
        b"data: {\"data\":{\"audio\":\"6162\",\"status\":1},\"trace_id\":\"interrupt-trace\"}\n\n",
    )]);
    interrupted.stream_error_after_chunks = Some("socket closed".into());
    let transport = MockTransport::new([interrupted]);
    let service = make_service(&transport);
    let mut stream = service
        .synthesize_stream(
            &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
            &credentials(),
        )
        .await
        .unwrap();
    assert!(stream.next_event().await.unwrap().is_some());
    match stream.next_event().await.unwrap_err() {
        MiniMaxTtsStreamError::Interrupted {
            bytes_delivered,
            request_id,
            ..
        } => {
            assert_eq!(bytes_delivered, 2);
            assert_eq!(request_id.as_deref(), Some("interrupt-trace"));
        }
        other => panic!("unexpected stream error: {other}"),
    }
    assert!(!stream.terminal_event_received());
}

#[tokio::test]
async fn http_tts_stream_rejects_malformed_audio_and_missing_audio() {
    for body in [
        &b"data: {\"data\":{\"audio\":\"0xz\",\"status\":1}}\n\n"[..],
        &b"data: [DONE]\n\n"[..],
    ] {
        let transport = MockTransport::new([sse_reply(vec![Bytes::copy_from_slice(body)])]);
        let service = make_service(&transport);
        let mut stream = service
            .synthesize_stream(
                &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
                &credentials(),
            )
            .await
            .unwrap();
        assert!(matches!(
            stream.next_event().await,
            Err(MiniMaxTtsStreamError::InvalidEvent { .. })
        ));
    }
}

#[tokio::test]
async fn malformed_base_response_cannot_make_stream_audio_look_successful() {
    let cases: [&[u8]; 2] = [
        b"data: {\"data\":{\"audio\":\"6162\",\"status\":1},\"base_resp\":{\"status_code\":\"1001\"}}\n\n",
        b"data: {\"data\":{\"audio\":\"6162\",\"status\":1},\"base_resp\":\"malformed\"}\n\n",
    ];
    for body in cases {
        let transport = MockTransport::new([sse_reply(vec![Bytes::copy_from_slice(body)])]);
        let service = make_service(&transport);
        let mut stream = service
            .synthesize_stream(
                &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
                &credentials(),
            )
            .await
            .unwrap();
        match stream.next_event().await.unwrap_err() {
            MiniMaxTtsStreamError::InvalidEvent {
                bytes_delivered,
                native,
                ..
            } => {
                assert_eq!(bytes_delivered, 0);
                assert!(native.is_some());
            }
            other => panic!("unexpected malformed-envelope error: {other}"),
        }
        assert!(!stream.terminal_event_received());
        assert!(stream.next_event().await.unwrap().is_none());
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn http_tts_stream_rejects_non_sse_success_and_preserves_http_errors() {
    let transport = MockTransport::new([
        reply(200, success("494433")),
        reply(
            503,
            json!({
                "trace_id": "stream-rejected",
                "base_resp": {"status_code": 2056, "status_msg": "quota unavailable"}
            }),
        ),
    ]);
    let service = make_service(&transport);
    let response_error = match service
        .synthesize_stream(
            &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
            &credentials(),
        )
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("expected a content-type validation error"),
    };
    match response_error {
        MiniMaxTtsError::InvalidResponse {
            request_id, native, ..
        } => {
            assert_eq!(request_id.as_deref(), Some("minimax-trace-1"));
            assert_eq!(native["data"]["audio"], "494433");
        }
        other => panic!("unexpected response error: {other}"),
    }

    let http_error = match service
        .synthesize_stream(
            &MiniMaxTtsRequest::new("Stream speech", "voice-123"),
            &credentials(),
        )
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("expected an HTTP rejection"),
    };
    match http_error {
        MiniMaxTtsError::Provider {
            http_status,
            code,
            request_id,
            dispatch,
            ..
        } => {
            assert_eq!(http_status, Some(503));
            assert_eq!(code, Some(2056));
            assert_eq!(request_id.as_deref(), Some("stream-rejected"));
            assert_eq!(dispatch, MiniMaxTtsDispatch::Rejected);
        }
        other => panic!("unexpected HTTP error: {other}"),
    }
}

#[tokio::test]
async fn http_tts_stream_validation_rejects_unsupported_wire_modes_before_http() {
    let transport = MockTransport::new([]);
    let service = make_service(&transport);

    let mut request = MiniMaxTtsRequest::new("Stream speech", "voice-123");
    request.audio_setting.format = MiniMaxTtsAudioFormat::Wav;
    assert!(matches!(
        service.synthesize_stream(&request, &credentials()).await,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));

    let mut request = MiniMaxTtsRequest::new("Stream speech", "voice-123");
    request.output_format = MiniMaxTtsOutputFormat::Url;
    assert!(matches!(
        service.synthesize_stream(&request, &credentials()).await,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));

    let mut request = MiniMaxTtsRequest::new("Stream speech", "voice-123");
    request.aigc_watermark = true;
    assert!(matches!(
        service.synthesize_stream(&request, &credentials()).await,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));

    let mut request = MiniMaxTtsRequest::new("Non-stream", "voice-123");
    request.subtitle_type = MiniMaxTtsSubtitleType::WordStreaming;
    assert!(matches!(
        service.synthesize(&request, &credentials()).await,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn scoped_voice_is_used_and_mismatched_references_fail_before_http() {
    let transport = MockTransport::new([reply(200, success("494433"))]);
    let active_service = make_service(&transport);
    let voice = MiniMaxVoiceRef::new(
        voice_reference(
            "minimax",
            "cn-voice",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::ChinaMainland,
        )
        .scope,
        MiniMaxVoiceKind::Cloned,
        "created-voice-17",
    )
    .unwrap();
    active_service
        .synthesize_with_voice(
            &MiniMaxTtsRequest::new("Use my created voice", "built-in-id"),
            &voice,
            &credentials(),
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&transport.requests.lock().unwrap()[0].body).unwrap();
    assert_eq!(body["voice_setting"]["voice_id"], "created-voice-17");

    let transport = MockTransport::new([]);
    let untouched_service = make_service(&transport);
    let mismatched = [
        voice_reference(
            "other-provider",
            "cn-voice",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::ChinaMainland,
        ),
        voice_reference(
            "minimax",
            "another-profile",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::ChinaMainland,
        ),
        voice_reference(
            "minimax",
            "cn-voice",
            "http://127.0.0.1:8418/v1",
            "another-account",
            MiniMaxVoicesRegion::ChinaMainland,
        ),
        voice_reference(
            "minimax",
            "cn-voice",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::International,
        ),
        voice_reference(
            "minimax",
            "cn-voice",
            "http://127.0.0.1:8419/v1",
            "team/account-17",
            MiniMaxVoicesRegion::ChinaMainland,
        ),
        voice_reference(
            "minimax",
            "cn-voice",
            "http://127.0.0.2:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::ChinaMainland,
        ),
    ];
    for voice in mismatched {
        assert!(matches!(
            untouched_service
                .synthesize_with_voice(
                    &MiniMaxTtsRequest::new("Must not be sent", "built-in-id"),
                    &voice,
                    &credentials(),
                )
                .await,
            Err(MiniMaxTtsError::InvalidRequest(_))
        ));
    }
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn url_mode_returns_expiring_provider_url_without_following_it() {
    let transport = MockTransport::new([reply(
        200,
        json!({
            "data": {"audio": "https://cdn.minimax.example/audio?id=signed", "status": 2},
            "trace_id": "trace-url",
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }),
    )]);
    let service = make_service(&transport);
    let mut request = MiniMaxTtsRequest::new("A temporary URL", "voice-abc");
    request.output_format = MiniMaxTtsOutputFormat::Url;
    let result = service.synthesize(&request, &credentials()).await.unwrap();
    let MiniMaxTtsOutput::Url(url) = result else {
        panic!("URL mode must retain the URL without downloading");
    };
    assert_eq!(url.url, "https://cdn.minimax.example/audio?id=signed");
    assert_eq!(url.valid_for, Duration::from_secs(24 * 60 * 60));
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
    let body: Value = serde_json::from_slice(&transport.requests.lock().unwrap()[0].body).unwrap();
    assert_eq!(body["output_format"], "url");
}

#[tokio::test]
async fn provider_level_error_is_rejected_even_when_http_status_is_success() {
    let transport = MockTransport::new([reply(
        200,
        json!({
            "base_resp": {"status_code": 2056, "status_msg": "quota unavailable"},
            "trace_id": "trace-rejected"
        }),
    )]);
    let error = make_service(&transport)
        .synthesize(&MiniMaxTtsRequest::new("Hello", "voice"), &credentials())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), MiniMaxTtsDispatch::Rejected);
    match error {
        MiniMaxTtsError::Provider {
            http_status,
            code,
            message,
            request_id,
            ..
        } => {
            assert_eq!(http_status, Some(200));
            assert_eq!(code, Some(2056));
            assert_eq!(message, "quota unavailable");
            assert_eq!(request_id.as_deref(), Some("trace-rejected"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn non_json_http_error_still_preserves_provider_rejection() {
    let mut response = reply(503, Value::Null);
    response.body = b"temporarily unavailable".to_vec();
    let transport = MockTransport::new([response]);
    let error = make_service(&transport)
        .synthesize(&MiniMaxTtsRequest::new("Hello", "voice"), &credentials())
        .await
        .unwrap_err();
    match error {
        MiniMaxTtsError::Provider {
            http_status,
            message,
            dispatch,
            ..
        } => {
            assert_eq!(http_status, Some(503));
            assert_eq!(message, "HTTP request rejected");
            assert_eq!(dispatch, MiniMaxTtsDispatch::Rejected);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn validation_rejects_invalid_options_before_send() {
    let transport = MockTransport::new([]);
    let service = make_service(&transport);
    let mut request = MiniMaxTtsRequest::new("hello", "voice");
    request.voice_setting.speed = Some(2.5);
    assert!(matches!(
        service.synthesize(&request, &credentials()).await,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());

    let mut request = MiniMaxTtsRequest::new("hello", "voice");
    request.audio_setting.format = MiniMaxTtsAudioFormat::Wav;
    request.audio_setting.channel = 3;
    assert!(matches!(
        service.synthesize(&request, &credentials()).await,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[test]
fn endpoint_and_account_scope_must_match_the_selected_region() {
    let transport = MockTransport::new([]);
    let wrong_region = MiniMaxTtsService::new(
        &transport,
        MiniMaxTtsConfig::new("profile", "account", MiniMaxTtsRegion::ChinaMainland)
            .with_endpoint("https://api.minimax.io/v1/t2a_v2"),
    );
    assert!(matches!(
        wrong_region,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));
    let missing_scope = MiniMaxTtsService::new(
        &transport,
        MiniMaxTtsConfig::new("profile", "  ", MiniMaxTtsRegion::International),
    );
    assert!(matches!(
        missing_scope,
        Err(MiniMaxTtsError::InvalidRequest(_))
    ));
    let international = MiniMaxTtsService::new(
        &transport,
        MiniMaxTtsConfig::new("profile", "account", MiniMaxTtsRegion::International)
            .with_endpoint("https://api-uw.minimax.io/v1/t2a_v2"),
    )
    .unwrap();
    assert_eq!(international.scope().provider_id.as_str(), "minimax");
}

#[test]
fn model_ids_have_the_exact_provider_spelling() {
    assert_eq!(MiniMaxTtsModel::Speech28Hd.as_str(), "speech-2.8-hd");
    assert_eq!(MiniMaxTtsModel::Speech02Turbo.as_str(), "speech-02-turbo");
}
