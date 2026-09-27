use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    transport::{HttpRequest, StreamResponse, Transport},
    vertex_speech::{
        VertexSpeechError, VertexSpeechModel, VertexSpeechRequest, VertexSpeechScope,
        VertexSpeechService, VertexSpeechSpeaker, VertexSpeechStreamError, VertexSpeechTurn,
    },
    RequestOptions,
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

fn scope(location: &str) -> VertexSpeechScope {
    VertexSpeechScope::new("vertex-tts", "vertex-prod", "demo-project", location).unwrap()
}

fn service(reply: Reply) -> (VertexSpeechService<'static>, &'static Mock) {
    let transport: &'static Mock = Box::leak(Box::new(Mock::new(reply)));
    (
        VertexSpeechService::new(transport, scope("us-central1")).unwrap(),
        transport,
    )
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("google-access-token".to_owned())),
        account_scope: Some("vertex-prod".into()),
        ..Default::default()
    }
}

fn headers(content_type: &str) -> Vec<(String, String)> {
    vec![
        ("content-type".into(), content_type.into()),
        ("x-goog-request-id".into(), "vertex-request-001".into()),
    ]
}

fn audio_response(audio: &[u8], mime_type: &str, finish_reason: &str) -> Value {
    json!({
        "candidates": [{
            "content": {"parts": [{
                "inlineData": {
                    "mimeType": mime_type,
                    "data": BASE64.encode(audio)
                }
            }]},
            "finishReason": finish_reason
        }],
        "modelVersion": "gemini-2.5-flash-tts"
    })
}

fn json_reply(value: Value) -> Reply {
    Reply::Response {
        status: 200,
        headers: headers("application/json"),
        body: Bytes::from(serde_json::to_vec(&value).unwrap()),
    }
}

fn request() -> VertexSpeechRequest {
    VertexSpeechRequest::single(
        VertexSpeechModel::Gemini25FlashTts,
        "Welcome to the demo.",
        "en-US",
        "Kore",
    )
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> &'a str {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
        .unwrap()
}

#[tokio::test]
async fn unary_request_uses_explicit_regional_vertex_route_and_decodes_pcm() {
    let pcm = [0_u8, 1, 255, 127];
    let (service, mock) = service(json_reply(audio_response(
        &pcm,
        "audio/L16;codec=pcm;rate=24000",
        "STOP",
    )));
    let response = service.synthesize(&request(), &options()).await.unwrap();

    assert_eq!(response.audio, Bytes::copy_from_slice(&pcm));
    assert_eq!(
        response.mime_type.as_deref(),
        Some("audio/L16;codec=pcm;rate=24000")
    );
    assert_eq!(response.pcm_sample_rate_hz, 24_000);
    assert_eq!(response.pcm_channels, 1);
    assert_eq!(response.pcm_bits_per_sample, 16);
    assert_eq!(response.finish_reason.as_deref(), Some("STOP"));
    assert_eq!(response.native["modelVersion"], "gemini-2.5-flash-tts");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert_eq!(sent.method, "POST");
    assert_eq!(
        sent.url,
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/demo-project/locations/us-central1/publishers/google/models/gemini-2.5-flash-tts:generateContent"
    );
    assert_eq!(
        header(&sent.headers, "authorization"),
        "Bearer google-access-token"
    );
    assert_eq!(header(&sent.headers, "x-goog-user-project"), "demo-project");
    assert_eq!(header(&sent.headers, "accept"), "application/json");

    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["contents"][0]["role"], "user");
    assert_eq!(
        body["contents"][0]["parts"][0]["text"],
        "Welcome to the demo."
    );
    assert_eq!(
        body["generationConfig"]["speechConfig"]["languageCode"],
        "en-US"
    );
    assert_eq!(
        body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
        "Kore"
    );
}

#[tokio::test]
async fn multi_speaker_request_encodes_labeled_contents_and_voice_config() {
    let (service, mock) = service(json_reply(audio_response(
        b"dialogue",
        "audio/pcm;rate=24000",
        "STOP",
    )));
    let request = VertexSpeechRequest::multi_speaker(
        VertexSpeechModel::Gemini25FlashTts,
        "en-US",
        vec![
            VertexSpeechSpeaker::new("Host", "Kore"),
            VertexSpeechSpeaker::new("Guest", "Puck"),
        ],
        vec![
            VertexSpeechTurn::new("Host", "Welcome."),
            VertexSpeechTurn::new("Guest", "Thank you."),
        ],
    )
    .with_prompt("A warm conversation");
    service.synthesize(&request, &options()).await.unwrap();

    let requests = mock.requests.lock().unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body["contents"][0]["parts"][0]["text"],
        "A warm conversation: Host: Welcome.\nGuest: Thank you."
    );
    assert_eq!(
        body["generationConfig"]["speechConfig"]["multiSpeakerVoiceConfig"]["speakerVoiceConfigs"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let voice_configs =
        &body["generationConfig"]["speechConfig"]["multiSpeakerVoiceConfig"]["speakerVoiceConfigs"];
    assert_eq!(voice_configs[0]["speaker"], "Host");
    assert_eq!(
        voice_configs[0]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
        "Kore"
    );
    assert_eq!(voice_configs[1]["speaker"], "Guest");
    assert_eq!(
        voice_configs[1]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
        "Puck"
    );
}

#[tokio::test]
async fn preflight_rejects_unsupported_model_region_and_scope_without_sending() {
    let (service, mock) = service(json_reply(audio_response(b"audio", "audio/pcm", "STOP")));
    let unsupported = VertexSpeechRequest::single(
        VertexSpeechModel::Gemini31FlashTtsPreview,
        "Hello.",
        "en-US",
        "Kore",
    );
    let error = service
        .synthesize(&unsupported, &options())
        .await
        .unwrap_err();
    assert!(matches!(error, VertexSpeechError::InvalidInput(_)));

    let mut mismatch = options();
    mismatch.account_scope = Some("different-account".into());
    let error = service.synthesize(&request(), &mismatch).await.unwrap_err();
    assert!(matches!(error, VertexSpeechError::ScopeMismatch));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn streaming_yields_sse_chunks_and_requires_a_terminal_finish_reason() {
    let first = audio_response(b"first-", "audio/L16;codec=pcm;rate=24000", "");
    let final_response = json!({
        "candidates": [{"content":{"parts":[]}, "finishReason":"MAX_TOKENS"}]
    });
    let wire = format!(
        "data: {}\n\ndata: {}",
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&final_response).unwrap()
    );
    let chunks = wire
        .as_bytes()
        .chunks(9)
        .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
        .collect();
    let mock = Box::leak(Box::new(Mock::new(Reply::Stream {
        status: 200,
        headers: headers("text/event-stream; charset=utf-8"),
        chunks,
    })));
    let service = VertexSpeechService::new(mock, scope("us-central1")).unwrap();
    let mut stream = service
        .synthesize_stream(&request(), &options())
        .await
        .unwrap();

    let first_event = stream.next_event().await.unwrap().unwrap();
    assert_eq!(first_event.audio, Bytes::from_static(b"first-"));
    assert_eq!(first_event.finish_reason, None);
    assert_eq!(
        first_event.native["candidates"][0]["content"]["parts"][0]["inlineData"]["data"],
        BASE64.encode(b"first-")
    );

    let final_event = stream.next_event().await.unwrap().unwrap();
    assert_eq!(final_event.audio, Bytes::new());
    assert_eq!(final_event.finish_reason.as_deref(), Some("MAX_TOKENS"));
    assert_eq!(stream.completion_reason(), Some("MAX_TOKENS"));
    assert!(stream.next_event().await.unwrap().is_none());

    let sent = &mock.requests.lock().unwrap()[0];
    assert!(sent.url.ends_with(":streamGenerateContent?alt=sse"));
    assert_eq!(header(&sent.headers, "accept"), "text/event-stream");
}

#[tokio::test]
async fn prompt_block_is_a_documented_terminal_without_audio() {
    let wire = format!(
        "data: {}\n\n",
        serde_json::to_string(&json!({
            "candidates": [],
            "promptFeedback": {"blockReason":"SAFETY"}
        }))
        .unwrap()
    );
    let mock = Box::leak(Box::new(Mock::new(Reply::Stream {
        status: 200,
        headers: headers("text/event-stream"),
        chunks: vec![Ok(Bytes::from(wire))],
    })));
    let service = VertexSpeechService::new(mock, scope("us-central1")).unwrap();
    let mut stream = service
        .synthesize_stream(&request(), &options())
        .await
        .unwrap();
    let event = stream.next_event().await.unwrap().unwrap();
    assert_eq!(event.audio, Bytes::new());
    assert_eq!(event.prompt_block_reason.as_deref(), Some("SAFETY"));
    assert_eq!(stream.completion_reason(), Some("PROMPT_BLOCKED:SAFETY"));
    assert!(stream.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn incompatible_response_mime_is_accepted_invalid_not_mislabeled_pcm() {
    let (service, mock) = service(json_reply(audio_response(b"wav", "audio/wav", "STOP")));
    let error = service
        .synthesize(&request(), &options())
        .await
        .unwrap_err();
    assert!(matches!(error, VertexSpeechError::InvalidResponse { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn clean_eof_without_finish_reason_is_reported_as_premature() {
    let wire = format!(
        "data: {}\n\n",
        serde_json::to_string(&audio_response(b"partial", "audio/pcm", "")).unwrap()
    );
    let mock = Box::leak(Box::new(Mock::new(Reply::Stream {
        status: 200,
        headers: headers("text/event-stream"),
        chunks: vec![Ok(Bytes::from(wire))],
    })));
    let service = VertexSpeechService::new(mock, scope("us-central1")).unwrap();
    let mut stream = service
        .synthesize_stream(&request(), &options())
        .await
        .unwrap();
    assert_eq!(
        stream.next_event().await.unwrap().unwrap().audio,
        Bytes::from_static(b"partial")
    );
    assert!(matches!(
        stream.next_event().await.unwrap_err(),
        VertexSpeechStreamError::PrematureEof {
            audio_bytes_received: 7,
            ..
        }
    ));
}
