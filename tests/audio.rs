use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{audio::*, protocol::*, *};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Sent {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    length: u64,
}
struct Mock {
    replies: Mutex<VecDeque<(u16, Vec<u8>)>>,
    sent: Mutex<Vec<Sent>>,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("audio must use request streaming")
    }
    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let mut body = Vec::new();
        let mut chunks = request.body;
        while let Some(chunk) = chunks.next().await {
            body.extend_from_slice(&chunk?);
        }
        self.sent.lock().unwrap().push(Sent {
            method: request.method,
            url: request.url,
            headers: request.headers,
            length: request.content_length,
            body,
        });
        let (status, body) = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected audio request");
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "req-audio".into())],
            body: stream::once(async move { Ok(Bytes::from(body)) }).boxed(),
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
fn setup(replies: Vec<(u16, Vec<u8>)>) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        sent: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}
fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("secret".into())),
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
fn known_speaker() -> KnownSpeakerReference {
    KnownSpeakerReference::new(
        "agent",
        "agent.wav",
        "audio/wav",
        2.5,
        b"agent sample".to_vec(),
    )
}
fn json_bytes(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}

#[tokio::test]
async fn streamed_transcription_preserves_native_events_and_requires_done() {
    let sse = b"data: {\"type\":\"transcript.text.delta\",\"delta\":\"bon\"}\n\ndata: {\"type\":\"transcript.text.done\",\"text\":\"bonjour\",\"languages\":[{\"code\":\"fr\"}]}\n\n";
    let (client, mock) = setup(vec![(200, sse.to_vec())]);
    let mut events = client
        .audio()
        .transcribe_stream(
            "openai",
            input(),
            &TranscriptionRequest::new(TranscriptionModel::GptTranscribe),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(events.request_id.as_deref(), Some("req-audio"));
    let delta = events.next_event().await.unwrap().unwrap();
    assert_eq!(delta.native["delta"], "bon");
    assert!(!delta.terminal);
    let done = events.next_event().await.unwrap().unwrap();
    assert_eq!(done.native["languages"][0]["code"], "fr");
    assert!(done.terminal);
    assert!(events.next_event().await.unwrap().is_none());
    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent[0].length, sent[0].body.len() as u64);
    assert!(String::from_utf8_lossy(&sent[0].body).contains("name=\"stream\"\r\n\r\ntrue\r\n"));
}

#[tokio::test]
async fn streamed_transcription_rejects_whisper_and_reports_missing_terminal() {
    let (client, mock) = setup(vec![(
        200,
        b"data: {\"type\":\"transcript.text.delta\",\"delta\":\"hi\"}\n\n".to_vec(),
    )]);
    let rejected = client
        .audio()
        .transcribe_stream(
            "openai",
            input(),
            &TranscriptionRequest::new(TranscriptionModel::Whisper1),
            &options(),
        )
        .await;
    assert!(matches!(
        rejected,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.sent.lock().unwrap().is_empty());
    let mut events = client
        .audio()
        .transcribe_stream(
            "openai",
            input(),
            &TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribe),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(
        events.next_event().await.unwrap().unwrap().native["delta"],
        "hi"
    );
    assert!(matches!(
        events.next_event().await,
        Err(TranscriptionStreamError::Interrupted { .. })
    ));
}

#[tokio::test]
async fn streamed_diarization_preserves_segments_and_provider_errors() {
    let sse = b"data: {\"type\":\"transcript.text.segment\",\"id\":\"seg_0\",\"start\":0.0,\"end\":1.0,\"text\":\"Hello\",\"speaker\":\"agent\"}\n\ndata: {\"type\":\"transcript.text.done\",\"text\":\"Hello\"}\n\n";
    let (client, mock) = setup(vec![
        (200, sse.to_vec()),
        (429, json_bytes(json!({"error":{"message":"rate limited"}}))),
    ]);
    let mut request = TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribeDiarize);
    request.format = AudioTextFormat::DiarizedJson;
    request.chunking_auto = true;
    request.known_speakers.push(known_speaker());
    let mut events = client
        .audio()
        .transcribe_stream("openai", input(), &request, &options())
        .await
        .unwrap();
    let segment = events.next_event().await.unwrap().unwrap();
    assert_eq!(segment.event_type, "transcript.text.segment");
    assert_eq!(segment.native["speaker"], "agent");
    assert!(events.next_event().await.unwrap().unwrap().terminal);
    assert!(String::from_utf8_lossy(&mock.sent.lock().unwrap()[0].body)
        .contains("name=\"chunking_strategy\"\r\n\r\nauto\r\n"));
    assert!(String::from_utf8_lossy(&mock.sent.lock().unwrap()[0].body)
        .contains("name=\"known_speaker_references[]\""));
    let result = client
        .audio()
        .transcribe_stream("openai", input(), &request, &options())
        .await;
    assert!(matches!(
        result,
        Err(AudioError::Provider { status: 429, .. })
    ));
}

#[tokio::test]
async fn streamed_transcription_rejects_invalid_json_event() {
    let (client, _) = setup(vec![(200, b"data: [DONE]\n\n".to_vec())]);
    let mut events = client
        .audio()
        .transcribe_stream(
            "openai",
            input(),
            &TranscriptionRequest::new(TranscriptionModel::GptTranscribe),
            &options(),
        )
        .await
        .unwrap();
    assert!(matches!(
        events.next_event().await,
        Err(TranscriptionStreamError::InvalidEvent { .. })
    ));
    assert!(events.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn whisper_transcription_streams_exact_multipart_and_decodes_timestamps() {
    let (client, mock) = setup(vec![(
        200,
        json_bytes(json!({"text":"hello","language":"en","duration":1.5,
        "words":[{"word":"hello","start":0.1,"end":0.9}],
        "segments":[{"text":"hello","start":0.0,"end":1.0,"speaker":"speaker_0"}]})),
    )]);
    let mut request = TranscriptionRequest::new(TranscriptionModel::Whisper1);
    request.format = AudioTextFormat::VerboseJson;
    request.timestamp_granularities = vec![TimestampGranularity::Word];
    let result = client
        .audio()
        .transcribe("openai", input(), &request, &options())
        .await
        .unwrap();
    assert_eq!(result.text, "hello");
    assert_eq!(result.words[0].start, 0.1);
    assert_eq!(result.segments[0].speaker.as_deref(), Some("speaker_0"));
    assert_eq!(result.request_id.as_deref(), Some("req-audio"));
    assert!(result.native.is_some());
    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(
        sent[0].url,
        "https://api.openai.com/v1/audio/transcriptions"
    );
    assert_eq!(sent[0].length, sent[0].body.len() as u64);
    assert!(sent[0].body.windows(4).any(|bytes| bytes == [0, 255, 1, 2]));
    let body = String::from_utf8_lossy(&sent[0].body);
    assert!(body.contains("name=\"model\"\r\n\r\nwhisper-1\r\n"));
    assert!(body.contains("name=\"timestamp_granularities[]\"\r\n\r\nword\r\n"));
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer secret"));
}

#[tokio::test]
async fn translation_uses_whisper_and_preserves_plain_text() {
    let (client, mock) = setup(vec![(200, b"Hello, world!".to_vec())]);
    let request = TranslationRequest {
        format: AudioTextFormat::Text,
        ..Default::default()
    };
    let result = client
        .audio()
        .translate("openai", input(), &request, &options())
        .await
        .unwrap();
    assert_eq!(result.text, "Hello, world!");
    assert!(result.native.is_none());
    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent[0].url, "https://api.openai.com/v1/audio/translations");
    assert!(String::from_utf8_lossy(&sent[0].body).contains("name=\"model\"\r\n\r\nwhisper-1\r\n"));
}

#[tokio::test]
async fn gpt_transcribe_preserves_detected_languages() {
    let (client, _) = setup(vec![(
        200,
        json_bytes(json!({"text":"bonjour", "languages":[{"code":"fr"}]})),
    )]);
    let result = client
        .audio()
        .transcribe(
            "openai",
            input(),
            &TranscriptionRequest::new(TranscriptionModel::GptTranscribe),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(result.languages, vec!["fr"]);
    assert_eq!(result.language, None);
}

#[tokio::test]
async fn diarized_json_keeps_speakers_and_optional_chunking() {
    let payload = json_bytes(
        json!({"text":"hello", "segments":[{"text":"hello","start":0.0,"end":1.0,"speaker":"agent"}]}),
    );
    let (client, mock) = setup(vec![(200, payload.clone()), (200, payload)]);
    let mut request = TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribeDiarize);
    request.format = AudioTextFormat::DiarizedJson;
    let short = client
        .audio()
        .transcribe("openai", input(), &request, &options())
        .await
        .unwrap();
    assert_eq!(short.segments[0].speaker.as_deref(), Some("agent"));
    request.chunking_auto = true;
    let long = client
        .audio()
        .transcribe("openai", input(), &request, &options())
        .await
        .unwrap();
    assert_eq!(long.segments[0].text, "hello");
    let sent = mock.sent.lock().unwrap();
    assert!(!String::from_utf8_lossy(&sent[0].body).contains("name=\"chunking_strategy\""));
    assert!(String::from_utf8_lossy(&sent[1].body)
        .contains("name=\"chunking_strategy\"\r\n\r\nauto\r\n"));
}

#[tokio::test]
async fn known_speaker_references_use_documented_multipart_data_urls() {
    let (client, mock) = setup(vec![(
        200,
        json_bytes(json!({
            "text":"Hello",
            "segments":[{"text":"Hello","start":0.0,"end":1.0,"speaker":"agent"}]
        })),
    )]);
    let mut request = TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribeDiarize);
    request.format = AudioTextFormat::DiarizedJson;
    request.known_speakers.push(known_speaker());

    let result = client
        .audio()
        .transcribe("openai", input(), &request, &options())
        .await
        .unwrap();
    assert_eq!(result.segments[0].speaker.as_deref(), Some("agent"));

    let requests = mock.sent.lock().unwrap();
    let body = String::from_utf8_lossy(&requests[0].body);
    let name_at = body.find("name=\"known_speaker_names[]\"").unwrap();
    let reference_at = body.find("name=\"known_speaker_references[]\"").unwrap();
    assert!(name_at < reference_at);
    assert!(body.contains("\r\n\r\nagent\r\n"));
    assert!(body.contains("data:audio/wav;base64,YWdlbnQgc2FtcGxl"));
}

#[tokio::test]
async fn unsupported_model_options_and_oversize_fail_before_http() {
    let (client, mock) = setup(vec![]);
    let mut request = TranscriptionRequest::new(TranscriptionModel::GptTranscribe);
    request.format = AudioTextFormat::VerboseJson;
    assert!(matches!(
        client
            .audio()
            .transcribe("openai", input(), &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    let request = TranscriptionRequest::new(TranscriptionModel::GptTranscribe)
        .with_known_speaker(known_speaker());
    assert!(matches!(
        client
            .audio()
            .transcribe("openai", input(), &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    let mut request = TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribeDiarize)
        .with_known_speaker(known_speaker());
    assert!(matches!(
        client
            .audio()
            .transcribe("openai", input(), &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    request.format = AudioTextFormat::DiarizedJson;
    request.known_speakers[0].duration_seconds = 1.99;
    assert!(matches!(
        client
            .audio()
            .transcribe("openai", input(), &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    let mut request = TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribeDiarize);
    request.format = AudioTextFormat::DiarizedJson;
    for index in 0..5 {
        let mut speaker = known_speaker();
        speaker.name = format!("speaker-{index}");
        request.known_speakers.push(speaker);
    }
    assert!(matches!(
        client
            .audio()
            .transcribe("openai", input(), &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    let mut request = TranscriptionRequest {
        model: TranscriptionModel::Gpt4oTranscribeDiarize,
        format: AudioTextFormat::DiarizedJson,
        ..TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribeDiarize)
    };
    request.prompt = Some("speaker hints".into());
    assert!(matches!(
        client
            .audio()
            .transcribe("openai", input(), &request, &options())
            .await,
        Err(AudioError::Llm(LlmError::InvalidRequest { .. }))
    ));
    let large = AudioInput {
        filename: "voice.wav".into(),
        media_type: "audio/wav".into(),
        size_bytes: 25_000_001,
        body: stream::empty().boxed(),
    };
    assert!(matches!(
        client
            .audio()
            .transcribe(
                "openai",
                large,
                &TranscriptionRequest::new(TranscriptionModel::GptTranscribe),
                &options()
            )
            .await,
        Err(AudioError::Llm(LlmError::RequestTooLarge { .. }))
    ));
    assert!(mock.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn declared_length_mismatch_fails_without_replaying_upload() {
    let (client, mock) = setup(vec![]);
    let short = AudioInput {
        filename: "voice.wav".into(),
        media_type: "audio/wav".into(),
        size_bytes: 5,
        body: stream::once(async { Ok(Bytes::from_static(b"four")) }).boxed(),
    };
    let error = client
        .audio()
        .transcribe(
            "openai",
            short,
            &TranscriptionRequest::new(TranscriptionModel::GptTranscribe),
            &options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        AudioError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert!(mock.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn provider_error_keeps_status_and_body() {
    let (client, _) = setup(vec![(
        429,
        json_bytes(json!({"error":{"message":"slow down"}})),
    )]);
    let error = client
        .audio()
        .translate(
            "openai",
            input(),
            &TranslationRequest::default(),
            &options(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, AudioError::Provider { status:429, body, .. } if body["error"]["message"] == "slow down")
    );
}

#[test]
fn route_pair_must_be_documented() {
    let mut p = profile();
    p.audio = ServiceSetting::Enabled(AudioRoute {
        transcriptions_endpoint: "https://api.openai.com/v1/audio/transcriptions".into(),
        translations_endpoint: "https://evil.example/v1/audio/translations".into(),
        speech_endpoint: None,
        auth: ServiceAuth::Bearer,
    });
    assert!(LlmClientBuilder::with_transport(
        Arc::new(Mock {
            replies: Mutex::new(VecDeque::new()),
            sent: Mutex::new(vec![])
        }),
        &[p]
    )
    .with_region(Region::International)
    .build()
    .is_err());
}
