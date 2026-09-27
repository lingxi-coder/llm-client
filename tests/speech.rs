use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{audio::*, protocol::*, *};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Reply {
    status: u16,
    chunks: Vec<Result<Bytes, LlmError>>,
}
struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected speech request");
        Ok(StreamResponse {
            status: reply.status,
            headers: vec![
                ("x-request-id".into(), "req-speech".into()),
                ("content-type".into(), "text/event-stream".into()),
            ],
            body: stream::iter(reply.chunks).boxed(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai", "profile_name":"openai", "base_url":"https://chat.example/v1",
        "protocol":"open_ai_responses", "auth":"none", "models":[],
        "audio":{"mode":"enabled","value":{
            "transcriptions_endpoint":"https://api.openai.com/v1/audio/transcriptions",
            "translations_endpoint":"https://api.openai.com/v1/audio/translations",
            "speech_endpoint":"https://api.openai.com/v1/audio/speech",
            "auth":{"type":"bearer"}}}
    }))
    .unwrap()
}
fn setup(replies: Vec<Reply>) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        requests: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}
fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("key".into())),
        account_scope: Some("openai-project".into()),
        ..Default::default()
    }
}
fn request() -> SpeechRequest {
    SpeechRequest {
        model: SpeechModel::Gpt4oMiniTts,
        input: "hello".into(),
        voice: SpeechVoice::Coral,
        format: SpeechFormat::Pcm,
        instructions: Some("warm tone".into()),
        speed: Some(1.25),
    }
}

#[tokio::test]
async fn pcm_audio_streams_binary_chunks_and_metadata() {
    let (client, mock) = setup(vec![Reply {
        status: 200,
        chunks: vec![
            Ok(Bytes::from_static(&[0, 255])),
            Ok(Bytes::from_static(&[1, 2])),
        ],
    }]);
    let mut speech = client
        .audio()
        .synthesize("openai", &request(), &options())
        .await
        .unwrap();
    assert_eq!(speech.format, SpeechFormat::Pcm);
    assert_eq!(speech.pcm_sample_rate_hz, Some(24_000));
    assert_eq!(speech.pcm_channels, Some(1));
    assert_eq!(speech.pcm_bits_per_sample, Some(16));
    assert_eq!(speech.request_id.as_deref(), Some("req-speech"));
    assert_eq!(
        speech.next_chunk().await.unwrap().unwrap().as_ref(),
        &[0, 255]
    );
    assert_eq!(
        speech.next_chunk().await.unwrap().unwrap().as_ref(),
        &[1, 2]
    );
    assert!(speech.next_chunk().await.unwrap().is_none());
    let sent = mock.requests.lock().unwrap();
    assert_eq!(sent[0].method, "POST");
    assert_eq!(sent[0].url, "https://api.openai.com/v1/audio/speech");
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "gpt-4o-mini-tts");
    assert_eq!(body["voice"], "coral");
    assert_eq!(body["response_format"], "pcm");
    assert_eq!(body["stream_format"], "audio");
    assert_eq!(body["speed"], 1.25);
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer key"));
}

#[tokio::test]
async fn approved_custom_voice_uses_native_object_and_legacy_rejects_it() {
    let (client, mock) = setup(vec![Reply {
        status: 200,
        chunks: vec![Ok(Bytes::from_static(b"audio"))],
    }]);
    let mut custom = request();
    let reference = client
        .audio()
        .voices()
        .import_approved_voice("openai", "voice_123abc", &options())
        .unwrap();
    custom.voice = SpeechVoice::Custom(reference.clone());
    let persisted = serde_json::to_value(&custom.voice).unwrap();
    assert_eq!(persisted["custom"]["voice_id"], "voice_123abc");
    assert_eq!(persisted["custom"]["account_scope"], "openai-project");
    assert_eq!(
        serde_json::from_value::<SpeechVoice>(persisted).unwrap(),
        custom.voice
    );
    let mut stream = client
        .audio()
        .synthesize("openai", &custom, &options())
        .await
        .unwrap();
    assert_eq!(
        stream.next_chunk().await.unwrap().unwrap().as_ref(),
        b"audio"
    );
    {
        let sent = mock.requests.lock().unwrap();
        let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
        assert_eq!(body["voice"], json!({"id":"voice_123abc"}));
        assert_ne!(
            serde_json::to_value(&custom).unwrap()["voice"],
            body["voice"]
        );
    }

    custom.model = SpeechModel::Tts1;
    custom.instructions = None;
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &custom, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    custom.model = SpeechModel::Gpt4oMiniTts;
    let invalid_id: CustomVoiceRef = serde_json::from_value(json!({
        "voice_id":"wrong",
        "provider_id":"openai",
        "profile_name":"openai",
        "endpoint_fingerprint":reference.endpoint_fingerprint(),
        "account_scope":"openai-project"
    }))
    .unwrap();
    custom.voice = SpeechVoice::Custom(invalid_id);
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &custom, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn custom_voice_scope_mismatches_fail_before_credentials_or_http() {
    let (client, mock) = setup(vec![]);
    let reference = client
        .audio()
        .voices()
        .import_approved_voice("openai", "voice_123abc", &options())
        .unwrap();
    let mut request = request();
    request.voice = SpeechVoice::Custom(reference.clone());

    let mut wrong_account_without_credentials = options();
    wrong_account_without_credentials.credential = None;
    wrong_account_without_credentials.account_scope = Some("different-project".into());
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &request, &wrong_account_without_credentials)
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));

    for (field, value) in [
        ("provider_id", json!("other-provider")),
        ("profile_name", json!("other-profile")),
        ("endpoint_fingerprint", json!("different-endpoint")),
        ("account_scope", json!("different-project")),
        ("voice_id", json!("invalid")),
    ] {
        let mut value_ref = serde_json::to_value(&reference).unwrap();
        value_ref[field] = value;
        let tampered: CustomVoiceRef = serde_json::from_value(value_ref).unwrap();
        request.voice = SpeechVoice::Custom(tampered);
        assert!(matches!(
            client
                .audio()
                .synthesize("openai", &request, &options())
                .await,
            Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
        ));
    }
    assert!(serde_json::from_value::<SpeechVoice>(json!({"id":"voice_123abc"})).is_err());
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn custom_voice_rejects_non_official_speech_route_before_http() {
    let (official_client, _) = setup(vec![]);
    let reference = official_client
        .audio()
        .voices()
        .import_approved_voice("openai", "voice_123abc", &options())
        .unwrap();
    let mut alternate = profile();
    if let ServiceSetting::Enabled(route) = &mut alternate.audio {
        route.transcriptions_endpoint = "https://proxy.invalid/v1/audio/transcriptions".into();
        route.translations_endpoint = "https://proxy.invalid/v1/audio/translations".into();
        route.speech_endpoint = Some("https://proxy.invalid/v1/audio/speech".into());
    }
    let mock = Arc::new(Mock {
        replies: Mutex::new(VecDeque::new()),
        requests: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[alternate])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut custom = request();
    custom.voice = SpeechVoice::Custom(reference);
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &custom, &options())
            .await,
        Err(AudioError::Llm(LlmError::UnsupportedCapability { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn interrupted_binary_stream_reports_delivered_bytes() {
    let (client, _) = setup(vec![Reply {
        status: 200,
        chunks: vec![
            Ok(Bytes::from_static(b"abc")),
            Err(LlmError::Transport {
                message: "lost".into(),
            }),
        ],
    }]);
    let mut speech = client
        .audio()
        .synthesize("openai", &request(), &options())
        .await
        .unwrap();
    let _ = speech.next_chunk().await.unwrap();
    let error = speech.next_chunk().await.unwrap_err();
    assert_eq!(error.bytes_delivered, 3);
    assert!(speech.next_chunk().await.unwrap().is_none());
}

#[tokio::test]
async fn legacy_voice_instructions_and_bad_speed_fail_before_http() {
    let (client, mock) = setup(vec![]);
    let mut request = request();
    request.model = SpeechModel::Tts1;
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    request.instructions = None;
    request.voice = SpeechVoice::Cedar;
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    request.voice = SpeechVoice::Alloy;
    request.speed = Some(5.0);
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    request.speed = None;
    request.input = "x".repeat(4097);
    assert!(matches!(
        client
            .audio()
            .synthesize("openai", &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn provider_error_remains_an_error_not_audio() {
    let body = serde_json::to_vec(&json!({"error":{"message":"quota"}})).unwrap();
    let (client, _) = setup(vec![Reply {
        status: 429,
        chunks: vec![Ok(body.into())],
    }]);
    let error = client
        .audio()
        .synthesize("openai", &request(), &options())
        .await
        .err()
        .unwrap();
    assert!(
        matches!(error, AudioError::Provider { status:429, body, .. } if body["error"]["message"] == "quota")
    );
}

#[tokio::test]
async fn successful_empty_speech_body_is_invalid() {
    let (client, _) = setup(vec![Reply {
        status: 200,
        chunks: vec![],
    }]);
    let mut speech = client
        .audio()
        .synthesize("openai", &request(), &options())
        .await
        .unwrap();
    let error = speech.next_chunk().await.unwrap_err();
    assert_eq!(error.bytes_delivered, 0);
    assert!(matches!(*error.source, LlmError::ProviderInternal { .. }));
}

#[tokio::test]
async fn sse_speech_decodes_audio_preserves_unknown_events_and_fuses_on_done() {
    let wire = concat!(
        "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AP8B\"}\n\n",
        "data: {\"type\":\"speech.audio.future\",\"custom\":{\"x\":1}}\n\n",
        "data: {\"type\":\"speech.audio.done\",\"usage\":{\"input_tokens\":2,\"output_tokens\":7,\"total_tokens\":9}}\n\n"
    )
    .as_bytes();
    let (client, mock) = setup(vec![Reply {
        status: 200,
        chunks: vec![
            Ok(Bytes::copy_from_slice(&wire[..17])),
            Ok(Bytes::copy_from_slice(&wire[17..68])),
            Ok(Bytes::copy_from_slice(&wire[68..])),
        ],
    }]);
    let mut speech = client
        .audio()
        .synthesize_stream("openai", &request(), &options())
        .await
        .unwrap();
    assert_eq!(speech.request_id.as_deref(), Some("req-speech"));

    let Some(SpeechEvent::AudioDelta {
        audio,
        raw_data,
        native,
    }) = speech.next_event().await.unwrap()
    else {
        panic!("expected decoded audio delta");
    };
    assert_eq!(audio.as_ref(), &[0, 255, 1]);
    assert_eq!(native["type"], "speech.audio.delta");
    assert_eq!(
        raw_data.as_ref(),
        br#"{"type":"speech.audio.delta","audio":"AP8B"}"#
    );

    let Some(SpeechEvent::Unknown {
        event_type,
        native,
        raw_data,
    }) = speech.next_event().await.unwrap()
    else {
        panic!("expected preserved unknown event");
    };
    assert_eq!(event_type.as_deref(), Some("speech.audio.future"));
    assert_eq!(native.as_ref().unwrap()["custom"]["x"], 1);
    assert!(raw_data.ends_with(b"\"x\":1}}"));

    let Some(event) = speech.next_event().await.unwrap() else {
        panic!("expected terminal done event");
    };
    assert!(event.is_terminal());
    let SpeechEvent::AudioDone { usage, .. } = event else {
        panic!("expected terminal done event");
    };
    assert_eq!(usage["total_tokens"], 9);
    assert!(speech.next_event().await.unwrap().is_none());

    let sent = mock.requests.lock().unwrap();
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "accept" && value == "text/event-stream"));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["stream_format"], "sse");
    assert_eq!(body["response_format"], "pcm");
}

#[tokio::test]
async fn undocumented_error_event_is_raw_and_eof_without_done_is_interrupted() {
    let raw = br#"{"type":"speech.error","message":"provider error"}"#;
    let wire = [b"data: ".as_slice(), raw, b"\n\n"].concat();
    let (client, _) = setup(vec![Reply {
        status: 200,
        chunks: vec![Ok(wire.into())],
    }]);
    let mut speech = client
        .audio()
        .synthesize_stream("openai", &request(), &options())
        .await
        .unwrap();
    assert!(matches!(
        speech.next_event().await.unwrap(),
        Some(SpeechEvent::Unknown {
            event_type: Some(event_type),
            raw_data,
            ..
        }) if event_type == "speech.error" && raw_data.as_ref() == raw
    ));
    assert!(matches!(
        speech.next_event().await,
        Err(SpeechEventStreamError::Interrupted {
            source: LlmError::StreamInterrupted { message }
        }) if message.contains("speech.audio.done")
    ));
    assert!(speech.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn sse_malformed_known_delta_is_a_fused_invalid_event() {
    let wire = concat!(
        "data: {\"type\":\"speech.audio.delta\",\"audio\":\"not-base64!\"}\n\n",
        "data: {\"type\":\"speech.audio.done\",\"usage\":{}}\n\n"
    );
    let (client, _) = setup(vec![Reply {
        status: 200,
        chunks: vec![Ok(Bytes::from_static(wire.as_bytes()))],
    }]);
    let mut speech = client
        .audio()
        .synthesize_stream("openai", &request(), &options())
        .await
        .unwrap();
    assert!(matches!(
        speech.next_event().await,
        Err(SpeechEventStreamError::InvalidEvent { message, .. })
            if message.contains("Base64")
    ));
    assert!(speech.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn sse_done_requires_the_documented_usage_shape() {
    let wire =
        "data: {\"type\":\"speech.audio.done\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}\n\n";
    let (client, _) = setup(vec![Reply {
        status: 200,
        chunks: vec![Ok(Bytes::from_static(wire.as_bytes()))],
    }]);
    let mut speech = client
        .audio()
        .synthesize_stream("openai", &request(), &options())
        .await
        .unwrap();
    assert!(matches!(
        speech.next_event().await,
        Err(SpeechEventStreamError::InvalidEvent { message, .. })
            if message.contains("usage.total_tokens")
    ));
    assert!(speech.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn sse_transport_interruption_and_http_errors_keep_their_classification() {
    let delta =
        Bytes::from_static(b"data: {\"type\":\"speech.audio.delta\",\"audio\":\"YQ==\"}\n\n");
    let (client, _) = setup(vec![Reply {
        status: 200,
        chunks: vec![
            Ok(delta),
            Err(LlmError::Transport {
                message: "lost connection".into(),
            }),
        ],
    }]);
    let mut speech = client
        .audio()
        .synthesize_stream("openai", &request(), &options())
        .await
        .unwrap();
    assert!(matches!(
        speech.next_event().await.unwrap(),
        Some(SpeechEvent::AudioDelta { audio, .. }) if audio.as_ref() == b"a"
    ));
    assert!(matches!(
        speech.next_event().await,
        Err(SpeechEventStreamError::Interrupted {
            source: LlmError::Transport { .. }
        })
    ));

    let body = serde_json::to_vec(&json!({"error":{"message":"quota"}})).unwrap();
    let (client, _) = setup(vec![Reply {
        status: 429,
        chunks: vec![Ok(body.into())],
    }]);
    assert!(matches!(
        client
            .audio()
            .synthesize_stream("openai", &request(), &options())
            .await,
        Err(AudioError::Provider { status: 429, body, .. }) if body["error"]["message"] == "quota"
    ));
}

#[tokio::test]
async fn legacy_models_are_rejected_before_sse_http_dispatch() {
    let (client, mock) = setup(vec![]);
    let mut request = request();
    request.model = SpeechModel::Tts1;
    request.instructions = None;
    request.voice = SpeechVoice::Alloy;
    assert!(matches!(
        client
            .audio()
            .synthesize_stream("openai", &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sse_rejects_non_official_gateways_before_http_dispatch() {
    let mut alternate = profile();
    if let ServiceSetting::Enabled(route) = &mut alternate.audio {
        route.transcriptions_endpoint = "https://proxy.invalid/v1/audio/transcriptions".into();
        route.translations_endpoint = "https://proxy.invalid/v1/audio/translations".into();
        route.speech_endpoint = Some("https://proxy.invalid/v1/audio/speech".into());
    }
    let mock = Arc::new(Mock {
        replies: Mutex::new(VecDeque::new()),
        requests: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[alternate])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(matches!(
        client
            .audio()
            .synthesize_stream("openai", &request(), &options())
            .await,
        Err(AudioError::Llm(LlmError::UnsupportedCapability { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn oversized_sse_frame_is_bounded_and_fuses_the_stream() {
    let mut wire = b"data: ".to_vec();
    wire.resize(8 * 1024 * 1024 + 1, b'x');
    let (client, _) = setup(vec![Reply {
        status: 200,
        chunks: vec![Ok(wire.into())],
    }]);
    let mut speech = client
        .audio()
        .synthesize_stream("openai", &request(), &options())
        .await
        .unwrap();
    assert!(matches!(
        speech.next_event().await,
        Err(SpeechEventStreamError::Interrupted {
            source: LlmError::StreamInterrupted { message }
        }) if message.contains("SSE event") && message.contains("limit")
    ));
    assert!(speech.next_event().await.unwrap().is_none());
}
