use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    gemini_speech::{
        GeminiSpeechDispatch, GeminiSpeechError, GeminiSpeechFormat, GeminiSpeechModel,
        GeminiSpeechRequest, GeminiSpeechSampleRate, GeminiSpeechScope, GeminiSpeechService,
        GeminiSpeechSpeaker, GeminiSpeechStreamError, GeminiSpeechTurn, GEMINI_SPEECH_ENDPOINT,
    },
    protocol::{LlmError, Secret},
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::sync::Mutex;

enum Reply {
    Response {
        status: u16,
        headers: Vec<(String, String)>,
        body: Bytes,
    },
    Stream {
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<Result<Bytes, LlmError>>,
    },
    Error(LlmError),
}

struct Mock {
    reply: Mutex<Option<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(reply: Reply) -> Self {
        Self {
            reply: Mutex::new(Some(reply)),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        match self.reply.lock().unwrap().take().expect("unexpected retry") {
            Reply::Error(error) => Err(error),
            Reply::Response {
                status,
                headers,
                body,
            } => Ok(StreamResponse {
                status,
                headers,
                body: stream::once(async move { Ok(body) }).boxed(),
            }),
            Reply::Stream {
                status,
                headers,
                chunks,
            } => Ok(StreamResponse {
                status,
                headers,
                body: stream::iter(chunks).boxed(),
            }),
        }
    }
}

fn service(reply: Reply) -> (GeminiSpeechService<'static>, &'static Mock) {
    let transport: &'static Mock = Box::leak(Box::new(Mock::new(reply)));
    let scope = GeminiSpeechScope::new(
        "gemini-voice",
        "google-project-prod",
        GEMINI_SPEECH_ENDPOINT,
    )
    .unwrap();
    (
        GeminiSpeechService::new(transport, scope).unwrap(),
        transport,
    )
}

fn json_response(value: Value) -> Reply {
    Reply::Response {
        status: 200,
        headers: vec![
            ("content-type".into(), "application/json".into()),
            ("x-goog-request-id".into(), "google-trace-001".into()),
        ],
        body: Bytes::from(serde_json::to_vec(&value).unwrap()),
    }
}

fn sse_frame(value: Value) -> Bytes {
    let event_type = value["event_type"].as_str().unwrap();
    Bytes::from(format!(
        "event: {event_type}\ndata: {}\n\n",
        serde_json::to_string(&value).unwrap()
    ))
}

fn audio_sse_reply(events: Vec<Value>, split: bool, done: bool) -> Reply {
    let mut wire = events
        .into_iter()
        .map(sse_frame)
        .flat_map(|frame| frame.to_vec())
        .collect::<Vec<_>>();
    if done {
        wire.extend_from_slice(b"event: done\ndata: [DONE]\n\n");
    }
    let chunks = if split && wire.len() > 2 {
        let split_at = wire.len() / 2;
        vec![
            Ok(Bytes::copy_from_slice(&wire[..split_at])),
            Ok(Bytes::copy_from_slice(&wire[split_at..])),
        ]
    } else {
        vec![Ok(Bytes::from(wire))]
    };
    Reply::Stream {
        status: 200,
        headers: vec![
            (
                "content-type".into(),
                "text/event-stream; charset=utf-8".into(),
            ),
            ("x-goog-request-id".into(), "google-stream-trace".into()),
        ],
        chunks,
    }
}

fn stream_events(audio: &[u8]) -> Vec<Value> {
    vec![
        json!({
            "event_type": "interaction.created",
            "event_id": "evt-created",
            "interaction": {"id": "v1_stream123", "status": "in_progress", "model": "gemini-3.8-flash-tts"}
        }),
        json!({"event_type":"step.start","event_id":"evt-start","index":0,"step":{"type":"model_output"}}),
        json!({
            "event_type":"step.delta","event_id":"evt-audio","index":0,
            "delta":{"type":"audio","data":BASE64.encode(audio),"mime_type":"audio/l16","sample_rate":24000,"channels":1}
        }),
        json!({"event_type":"step.stop","event_id":"evt-stop","index":0}),
        json!({
            "event_type":"interaction.completed","event_id":"evt-completed",
            "interaction":{"id":"v1_stream123","status":"completed","usage":{"total_tokens":15}}
        }),
    ]
}

fn interaction_response(audio: &[u8]) -> Value {
    json!({
        "id": "int_123",
        "object": "interaction",
        "status": "completed",
        "model": "gemini-3.8-flash-tts",
        "steps": [{
            "type": "model_output",
            "content": [{
                "type": "audio",
                "data": BASE64.encode(audio),
                "mime_type": "audio/wav",
                "sample_rate": 24000,
            }]
        }],
        "usage": {"total_tokens": 23}
    })
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> &'a str {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
        .unwrap()
}

#[tokio::test]
async fn synthesizes_with_documented_interactions_request_and_retains_native_response() {
    let expected_audio = b"RIFF\x24\0\0\0WAVE";
    let (svc, mock) = service(json_response(interaction_response(expected_audio)));
    let request = GeminiSpeechRequest::new(
        GeminiSpeechModel::Gemini38FlashTts,
        "Have a wonderful day!",
        "Kore",
    )
    .unwrap()
    .with_style("cheerful and friendly")
    .unwrap();
    let response = svc
        .synthesize(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();

    assert_eq!(response.audio, Bytes::from_static(expected_audio));
    assert_eq!(response.requested_format, GeminiSpeechFormat::Wav);
    assert_eq!(
        response.requested_sample_rate,
        GeminiSpeechSampleRate::Hz24000
    );
    assert_eq!(response.response_mime_type.as_deref(), Some("audio/wav"));
    assert_eq!(response.response_sample_rate, Some(24000));
    assert_eq!(response.interaction_id.as_deref(), Some("int_123"));
    assert_eq!(response.model.as_deref(), Some("gemini-3.8-flash-tts"));
    assert_eq!(response.scope.account_scope(), "google-project-prod");
    assert_eq!(response.native["usage"]["total_tokens"], 23);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.url, GEMINI_SPEECH_ENDPOINT);
    assert_eq!(header(&sent.headers, "x-goog-api-key"), "gemini-key");
    assert_eq!(header(&sent.headers, "content-type"), "application/json");
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["model"], "gemini-3.8-flash-tts");
    assert_eq!(body["stream"], false);
    assert_eq!(body["input"][0]["type"], "user_input");
    assert_eq!(body["input"][0]["content"][0]["type"], "text");
    assert_eq!(
        body["input"][0]["content"][0]["text"],
        "Have a wonderful day!"
    );
    assert_eq!(
        body["input"][0]["content"][0]["annotations"][0]["type"],
        "speech_metadata"
    );
    assert_eq!(
        body["input"][0]["content"][0]["annotations"][0]["style"],
        "cheerful and friendly"
    );
    assert_eq!(body["response_format"]["type"], "audio");
    assert_eq!(body["response_format"]["mime_type"], "audio/wav");
    assert_eq!(body["response_format"]["sample_rate"], 24000);
    assert_eq!(
        body["generation_config"]["speech_config"][0]["voice"],
        "Kore"
    );
}

#[tokio::test]
async fn supports_documented_pcm_and_sample_rate_configuration() {
    let pcm = [0_u8, 255, 1, 128];
    let (svc, mock) = service(json_response(json!({
        "id": "int_pcm",
        "model": "gemini-3.8-flash-lite-tts",
        "steps": [{"type":"model_output","content":[{
            "type":"audio","data":BASE64.encode(pcm),"mime_type":"audio/l16","sample_rate":8000
        }]}]
    })));
    let request =
        GeminiSpeechRequest::new(GeminiSpeechModel::Gemini38FlashLiteTts, "Hello.", "Puck")
            .unwrap()
            .with_format(GeminiSpeechFormat::LinearPcm16)
            .with_sample_rate(GeminiSpeechSampleRate::Hz8000);
    let response = svc
        .synthesize(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();
    assert_eq!(response.audio, Bytes::copy_from_slice(&pcm));
    assert_eq!(response.response_mime_type.as_deref(), Some("audio/l16"));
    let body: Value = serde_json::from_slice(&mock.requests.lock().unwrap()[0].body).unwrap();
    assert_eq!(body["model"], "gemini-3.8-flash-lite-tts");
    assert_eq!(body["response_format"]["mime_type"], "audio/l16");
    assert_eq!(body["response_format"]["sample_rate"], 8000);
}

#[test]
fn rejects_non_google_endpoint_and_invalid_inputs_before_transport() {
    assert!(GeminiSpeechScope::new(
        "profile",
        "account",
        "https://example.com/v1beta/interactions"
    )
    .is_err());
    assert!(GeminiSpeechScope::new(
        "profile",
        "account",
        "https://generativelanguage.googleapis.com/v1beta/interactions?key=inline"
    )
    .is_err());
    assert!(GeminiSpeechRequest::new(GeminiSpeechModel::Gemini38FlashTts, "\0", "Kore").is_err());
    assert!(GeminiSpeechRequest::new(
        GeminiSpeechModel::Gemini38FlashTts,
        "Hello",
        "voice with spaces"
    )
    .is_err());
    assert!(
        serde_json::from_value::<GeminiSpeechModel>(json!("gemini-3.1-flash-tts-preview")).is_err()
    );
}

#[tokio::test]
async fn transport_failure_has_unknown_dispatch_and_is_not_retried() {
    let (svc, mock) = service(Reply::Error(LlmError::Transport {
        message: "connection closed".into(),
    }));
    let request =
        GeminiSpeechRequest::new(GeminiSpeechModel::Gemini38FlashTts, "Hello.", "Kore").unwrap();
    let error = svc
        .synthesize(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GeminiSpeechDispatch::Unknown);
    assert!(matches!(error, GeminiSpeechError::OutcomeUnknown { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn provider_client_error_is_reported_as_rejected() {
    let mock = Box::leak(Box::new(Mock::new(Reply::Response {
        status: 400,
        headers: vec![("content-type".into(), "application/json".into())],
        body: Bytes::from_static(br#"{"error":{"message":"unsupported voice"}}"#),
    })));
    let scope = GeminiSpeechScope::new("profile", "account", GEMINI_SPEECH_ENDPOINT).unwrap();
    let svc = GeminiSpeechService::new(mock, scope).unwrap();
    let request =
        GeminiSpeechRequest::new(GeminiSpeechModel::Gemini38FlashTts, "Hello.", "Kore").unwrap();
    let error = svc
        .synthesize(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GeminiSpeechDispatch::Rejected);
    assert!(matches!(
        error,
        GeminiSpeechError::Provider {
            status: 400,
            ref message,
            ..
        } if message == "unsupported voice"
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn success_without_audio_is_an_accepted_invalid_response() {
    let (svc, _) = service(json_response(json!({
        "id": "int_no_audio",
        "model": "gemini-3.8-flash-tts",
        "steps": [{"type":"model_output","content":[{"type":"text","text":"not audio"}]}]
    })));
    let request =
        GeminiSpeechRequest::new(GeminiSpeechModel::Gemini38FlashTts, "Hello.", "Kore").unwrap();
    let error = svc
        .synthesize(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GeminiSpeechDispatch::Accepted);
    assert!(matches!(error, GeminiSpeechError::InvalidResponse { .. }));
}

#[tokio::test]
async fn streams_documented_audio_deltas_and_retains_native_events() {
    let pcm = b"pcm-chunk";
    let (svc, mock) = service(audio_sse_reply(stream_events(pcm), true, true));
    let request = GeminiSpeechRequest::new_streaming(
        GeminiSpeechModel::Gemini38FlashTts,
        "Have a wonderful day!",
        "Kore",
    )
    .unwrap()
    .with_style("cheerful and friendly")
    .unwrap();
    let mut response = svc
        .synthesize_stream(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();

    assert_eq!(response.request_id(), Some("google-stream-trace"));
    assert_eq!(response.requested_format(), GeminiSpeechFormat::LinearPcm16);
    assert_eq!(
        response.requested_sample_rate(),
        GeminiSpeechSampleRate::Hz24000
    );
    let mut events = Vec::new();
    let mut audio = Vec::new();
    while let Some(event) = response.next_event().await.unwrap() {
        if let Some(chunk) = event.audio_chunk.as_ref() {
            audio.extend_from_slice(chunk);
            assert_eq!(event.audio_mime_type.as_deref(), Some("audio/l16"));
            assert_eq!(event.audio_sample_rate, Some(24000));
            assert_eq!(event.audio_channels, Some(1));
        }
        events.push(event);
    }
    assert_eq!(audio, pcm);
    assert_eq!(events.len(), 5);
    assert_eq!(events[2].event_type, "step.delta");
    assert_eq!(events[2].native["delta"]["data"], BASE64.encode(pcm));
    assert_eq!(events[2].audio_chunk, Some(Bytes::from_static(pcm)));
    assert_eq!(events[4].native["interaction"]["usage"]["total_tokens"], 15);
    assert_eq!(response.interaction_id(), Some("v1_stream123"));
    assert_eq!(
        response.reference().unwrap().account_scope(),
        "google-project-prod"
    );
    assert_eq!(response.last_event_id(), Some("evt-completed"));
    assert_eq!(response.audio_bytes_received(), pcm.len());

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert_eq!(header(&sent.headers, "accept"), "text/event-stream");
    assert_eq!(header(&sent.headers, "x-goog-api-key"), "gemini-key");
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["response_format"]["mime_type"], "audio/l16");
    assert_eq!(body["response_format"]["sample_rate"], 24000);
}

#[tokio::test]
async fn interruption_reports_scoped_interaction_cursor_and_partial_audio_without_retry() {
    let mut chunks = stream_events(b"first")[..3]
        .iter()
        .cloned()
        .map(sse_frame)
        .map(Ok)
        .collect::<Vec<_>>();
    chunks.push(Err(LlmError::Transport {
        message: "connection reset".into(),
    }));
    let (svc, mock) = service(Reply::Stream {
        status: 200,
        headers: vec![("content-type".into(), "text/event-stream".into())],
        chunks,
    });
    let request =
        GeminiSpeechRequest::new_streaming(GeminiSpeechModel::Gemini38FlashTts, "Hello.", "Kore")
            .unwrap();
    let mut response = svc
        .synthesize_stream(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();
    assert_eq!(
        response.next_event().await.unwrap().unwrap().event_type,
        "interaction.created"
    );
    assert_eq!(
        response.next_event().await.unwrap().unwrap().event_type,
        "step.start"
    );
    assert_eq!(
        response.next_event().await.unwrap().unwrap().audio_chunk,
        Some(Bytes::from_static(b"first"))
    );
    let error = response.next_event().await.unwrap_err();
    assert!(matches!(
        error,
        GeminiSpeechStreamError::Interrupted {
            interaction_id: Some(ref id),
            last_event_id: Some(ref cursor),
            audio_bytes_received: 5,
            interaction_completed: false,
            ..
        } if id == "v1_stream123" && cursor == "evt-audio"
    ));
    assert_eq!(response.reference().unwrap().profile_name(), "gemini-voice");
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn stream_error_event_preserves_native_error_and_is_not_retried() {
    let reply = audio_sse_reply(
        vec![json!({
            "event_type":"error","event_id":"evt-error",
            "error":{"code":"gateway_timeout","message":"deadline expired"}
        })],
        false,
        false,
    );
    let (svc, mock) = service(reply);
    let request = GeminiSpeechRequest::new_streaming(
        GeminiSpeechModel::Gemini38FlashLiteTts,
        "Hello.",
        "Kore",
    )
    .unwrap();
    let mut response = svc
        .synthesize_stream(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();
    let error = response.next_event().await.unwrap_err();
    assert!(matches!(
        error,
        GeminiSpeechStreamError::ProviderEvent {
            ref message,
            code: Some(ref code),
            event_id: Some(ref event_id),
            ref native,
            ..
        } if message == "deadline expired"
            && code == "gateway_timeout"
            && event_id == "evt-error"
            && native["event_type"] == "error"
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn completion_without_done_marker_is_reported_as_interrupted() {
    let (svc, _) = service(audio_sse_reply(stream_events(b"ok"), false, false));
    let request =
        GeminiSpeechRequest::new_streaming(GeminiSpeechModel::Gemini38FlashTts, "Hello.", "Kore")
            .unwrap();
    let mut response = svc
        .synthesize_stream(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();
    loop {
        match response.next_event().await {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("missing [DONE] must not complete a stream"),
            Err(GeminiSpeechStreamError::Interrupted {
                interaction_completed,
                interaction_id,
                last_event_id,
                ..
            }) => {
                assert!(interaction_completed);
                assert_eq!(interaction_id.as_deref(), Some("v1_stream123"));
                assert_eq!(last_event_id.as_deref(), Some("evt-completed"));
                break;
            }
            Err(other) => panic!("unexpected stream error: {other}"),
        }
    }
}

#[tokio::test]
async fn multi_speaker_unary_encodes_turn_metadata_and_conversational_voice_map() {
    let (svc, mock) = service(json_response(interaction_response(b"multi-speaker-audio")));
    let request = GeminiSpeechRequest::new_multi_speaker(
        GeminiSpeechModel::Gemini38FlashTts,
        vec![
            GeminiSpeechSpeaker::new("Host", "Kore"),
            GeminiSpeechSpeaker::new("Guest", "Puck"),
        ],
        vec![
            GeminiSpeechTurn::new("Host", "Welcome to the show.")
                .with_style("warm and conversational"),
            GeminiSpeechTurn::new("Guest", "Thanks for having me."),
        ],
    )
    .unwrap();
    let response = svc
        .synthesize(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();
    assert_eq!(response.audio, Bytes::from_static(b"multi-speaker-audio"));

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    let content = &body["input"][0]["content"];
    assert_eq!(content.as_array().unwrap().len(), 2);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Welcome to the show.");
    assert_eq!(content[0]["annotations"][0]["type"], "speech_metadata");
    assert_eq!(content[0]["annotations"][0]["speaker"], "Host");
    assert_eq!(
        content[0]["annotations"][0]["style"],
        "warm and conversational"
    );
    assert_eq!(content[1]["text"], "Thanks for having me.");
    assert_eq!(content[1]["annotations"][0]["speaker"], "Guest");
    let speech_config = &body["generation_config"]["speech_config"];
    assert_eq!(speech_config["mode"], "conversational");
    assert_eq!(speech_config["speakers"].as_array().unwrap().len(), 2);
    assert_eq!(speech_config["speakers"][0]["speaker"], "Host");
    assert_eq!(speech_config["speakers"][0]["voice"], "Kore");
    assert_eq!(speech_config["speakers"][1]["speaker"], "Guest");
    assert_eq!(speech_config["speakers"][1]["voice"], "Puck");
    assert!(body["generation_config"]["speech_config"]["speakers"].is_array());
}

#[test]
fn multi_speaker_construction_rejects_ambiguous_labels_and_empty_turns() {
    let speakers = || {
        vec![
            GeminiSpeechSpeaker::new("A", "Kore"),
            GeminiSpeechSpeaker::new("B", "Puck"),
        ]
    };
    let turns = || {
        vec![
            GeminiSpeechTurn::new("A", "Hello."),
            GeminiSpeechTurn::new("B", "Hi."),
        ]
    };

    assert!(GeminiSpeechRequest::new_multi_speaker(
        GeminiSpeechModel::Gemini38FlashTts,
        vec![GeminiSpeechSpeaker::new("A", "Kore")],
        turns(),
    )
    .is_err());
    assert!(GeminiSpeechRequest::new_multi_speaker(
        GeminiSpeechModel::Gemini38FlashTts,
        vec![
            GeminiSpeechSpeaker::new("A", "Kore"),
            GeminiSpeechSpeaker::new("A", "Puck"),
        ],
        turns(),
    )
    .is_err());
    assert!(GeminiSpeechRequest::new_multi_speaker(
        GeminiSpeechModel::Gemini38FlashTts,
        speakers(),
        vec![GeminiSpeechTurn::new("Unknown", "Hello.")],
    )
    .is_err());
    assert!(GeminiSpeechRequest::new_multi_speaker(
        GeminiSpeechModel::Gemini38FlashTts,
        speakers(),
        Vec::new(),
    )
    .is_err());
}

#[tokio::test]
async fn multi_speaker_streaming_uses_pcm_default_and_retains_native_turns() {
    let (svc, mock) = service(audio_sse_reply(stream_events(b"dialogue-pcm"), true, true));
    let request = GeminiSpeechRequest::new_multi_speaker_streaming(
        GeminiSpeechModel::Gemini38FlashTts,
        vec![
            GeminiSpeechSpeaker::new("Host", "Kore"),
            GeminiSpeechSpeaker::new("Guest", "Puck"),
        ],
        vec![
            GeminiSpeechTurn::new("Host", "Welcome."),
            GeminiSpeechTurn::new("Guest", "Thank you."),
        ],
    )
    .unwrap();
    let mut response = svc
        .synthesize_stream(&Secret::new("gemini-key".to_owned()), &request)
        .await
        .unwrap();
    let mut audio = Vec::new();
    while let Some(event) = response.next_event().await.unwrap() {
        if let Some(chunk) = event.audio_chunk {
            audio.extend_from_slice(&chunk);
        }
    }
    assert_eq!(audio.as_slice(), b"dialogue-pcm");
    let sent = mock.requests.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["response_format"]["mime_type"], "audio/l16");
    assert_eq!(body["response_format"]["sample_rate"], 24000);
    assert_eq!(
        body["generation_config"]["speech_config"]["mode"],
        "conversational"
    );
    assert_eq!(
        body["generation_config"]["speech_config"]["speakers"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}
