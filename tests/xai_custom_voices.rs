use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    audio::AudioInput,
    protocol::{LlmError, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
    xai_audio::{
        XaiAudioConfig, XaiAudioCredentials, XaiAudioError, XaiAudioService, XaiCustomVoiceAge,
        XaiCustomVoiceCreateRequest, XaiCustomVoiceGender, XaiCustomVoiceListRequest,
        XaiCustomVoicePatch, XaiCustomVoiceRef, XaiCustomVoiceTone, XaiCustomVoiceUseCase,
    },
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    chunks: Vec<Bytes>,
}

enum Outcome {
    Response(Reply),
    Transport(LlmError),
}

struct SentRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    content_length: Option<u64>,
}

struct MockTransport {
    outcomes: Mutex<VecDeque<Outcome>>,
    sent: Mutex<Vec<SentRequest>>,
}

struct DropSentinel(Arc<AtomicBool>);

impl Drop for DropSentinel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct DropSentinelTransport {
    chunks: Mutex<Option<Vec<Result<Bytes, LlmError>>>>,
    dropped: Arc<AtomicBool>,
}

impl DropSentinelTransport {
    fn new(chunks: Vec<Result<Bytes, LlmError>>) -> Self {
        Self {
            chunks: Mutex::new(Some(chunks)),
            dropped: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl Transport for DropSentinelTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let chunks = self
            .chunks
            .lock()
            .unwrap()
            .take()
            .expect("the custom voice audio request is sent once");
        let guard = DropSentinel(self.dropped.clone());
        let body = stream::unfold((chunks.into_iter(), guard), |(mut chunks, guard)| async move {
            chunks.next().map(|chunk| (chunk, (chunks, guard)))
        })
        .boxed();
        Ok(StreamResponse {
            status: 200,
            headers: vec![("content-type".into(), "audio/wav".into())],
            body,
        })
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "drop sentinel only serves a regular GET request".into(),
        })
    }
}

impl MockTransport {
    fn new(outcomes: impl IntoIterator<Item = Outcome>) -> Self {
        Self {
            outcomes: Mutex::new(outcomes.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
        }
    }

    fn response(reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: stream::iter(reply.chunks.into_iter().map(Ok)).boxed(),
        }
    }

    fn next_response(&self) -> Result<StreamResponse, LlmError> {
        match self
            .outcomes
            .lock()
            .unwrap()
            .pop_front()
            .expect("an HTTP outcome is queued")
        {
            Outcome::Response(reply) => Ok(Self::response(reply)),
            Outcome::Transport(error) => Err(error),
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
            content_length: Some(request.body.len() as u64),
            body: request.body.to_vec(),
        });
        self.next_response()
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let mut body = Vec::new();
        let mut input = request.body;
        while let Some(chunk) = input.next().await {
            body.extend_from_slice(&chunk?);
        }
        self.sent.lock().unwrap().push(SentRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body,
            content_length: Some(request.content_length),
        });
        self.next_response()
    }
}

fn json_reply(status: u16, headers: Vec<(String, String)>, value: Value) -> Outcome {
    Outcome::Response(Reply {
        status,
        headers,
        chunks: vec![Bytes::from(serde_json::to_vec(&value).unwrap())],
    })
}

fn audio_reply(content_type: &str, chunks: impl IntoIterator<Item = &'static [u8]>) -> Outcome {
    Outcome::Response(Reply {
        status: 200,
        headers: vec![("content-type".into(), content_type.into())],
        chunks: chunks.into_iter().map(Bytes::from_static).collect(),
    })
}

fn credentials(key: &str) -> XaiAudioCredentials {
    XaiAudioCredentials::new(Secret::new(key.to_owned()))
}

fn service(transport: &dyn Transport) -> XaiAudioService<'_> {
    XaiAudioService::new(
        transport,
        XaiAudioConfig::new("work", "team/account-7")
            .with_api_base_url("http://127.0.0.1:8391/v1")
            .with_request_timeout(Duration::from_secs(2)),
    )
    .unwrap()
}

fn voice(id: &str) -> Value {
    json!({
        "voice_id": id,
        "name": "Friendly Narrator",
        "description": "Warm narration",
        "gender": "female",
        "accent": "American",
        "age": "young",
        "language": "en-US",
        "use_case": "narration",
        "tone": "warm",
        "created_at": "2026-04-26T18:56:34.872993+00:00",
        "provider_extension": {"future": true}
    })
}

#[tokio::test]
async fn create_custom_voice_streams_documented_multipart_and_preserves_native_metadata() {
    let returned = voice("nlbqfwie");
    let transport = MockTransport::new([json_reply(
        201,
        vec![("x-request-id".into(), "create-voice-1".into())],
        returned.clone(),
    )]);
    let service = service(&transport);
    let request = XaiCustomVoiceCreateRequest::new(AudioInput::from_bytes(
        "reference.wav",
        "audio/wav",
        Bytes::from_static(b"audio-sample"),
    ))
    .with_duration_seconds(100.0)
    .with_name("Friendly Narrator")
    .with_description("Warm narration")
    .with_gender(XaiCustomVoiceGender::Female)
    .with_accent("American")
    .with_age(XaiCustomVoiceAge::Young)
    .with_language("en-US")
    .with_use_case(XaiCustomVoiceUseCase::Narration)
    .with_tone(XaiCustomVoiceTone::Warm);
    let created = service
        .create_custom_voice(request, &credentials("xai-key"))
        .await
        .unwrap();

    assert_eq!(created.reference.scope().account_scope, "team/account-7");
    assert_eq!(created.reference.voice_id(), "nlbqfwie");
    assert_eq!(created.name.as_deref(), Some("Friendly Narrator"));
    assert_eq!(
        created.created_at.as_deref(),
        Some("2026-04-26T18:56:34.872993+00:00")
    );
    assert_eq!(created.request_id.as_deref(), Some("create-voice-1"));
    assert_eq!(created.native, returned);

    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    let call = &sent[0];
    assert_eq!(call.method, "POST");
    assert_eq!(call.url, "http://127.0.0.1:8391/v1/custom-voices");
    assert_eq!(call.content_length, Some(call.body.len() as u64));
    assert!(call.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer xai-key"
    }));
    let content_type = call
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value)
        .unwrap();
    assert!(content_type.starts_with("multipart/form-data; boundary="));
    let boundary = content_type.split_once("boundary=").unwrap().1;
    let multipart = String::from_utf8_lossy(&call.body);
    for expected in [
        "name=\"name\"\r\n\r\nFriendly Narrator",
        "name=\"description\"\r\n\r\nWarm narration",
        "name=\"gender\"\r\n\r\nfemale",
        "name=\"accent\"\r\n\r\nAmerican",
        "name=\"age\"\r\n\r\nyoung",
        "name=\"language\"\r\n\r\nen-US",
        "name=\"use_case\"\r\n\r\nnarration",
        "name=\"tone\"\r\n\r\nwarm",
        "filename=\"reference.wav\"\r\nContent-Type: audio/wav\r\n\r\naudio-sample",
    ] {
        assert!(
            multipart.contains(expected),
            "missing multipart content {expected:?}"
        );
    }
    assert!(multipart.ends_with(&format!("\r\n--{boundary}--\r\n")));
}

#[tokio::test]
async fn list_custom_voices_scopes_rows_and_encodes_pagination_query() {
    let first = voice("nlbqfwie");
    let second = voice("evemusic");
    let transport = MockTransport::new([json_reply(
        200,
        vec![("request-id".into(), "list-voice-1".into())],
        json!({
            "voices": [first.clone(), second],
            "pagination_token": "cursor /+=",
            "provider_extension": "preserved"
        }),
    )]);
    let voice_service = service(&transport);
    let page = voice_service
        .list_custom_voices(
            &XaiCustomVoiceListRequest::new().with_limit(2),
            &credentials("xai-key"),
        )
        .await
        .unwrap();
    assert_eq!(page.voices.len(), 2);
    assert_eq!(page.voices[0].native, first);
    assert_eq!(
        page.voices[0].reference.scope().account_scope,
        "team/account-7"
    );
    assert_eq!(
        page.next_page.as_ref().unwrap().scope().account_scope,
        "team/account-7"
    );
    assert_eq!(page.native["provider_extension"], "preserved");
    assert_eq!(page.request_id.as_deref(), Some("list-voice-1"));

    let next = page.next_page.unwrap();
    let next_transport = MockTransport::new([json_reply(
        200,
        Vec::new(),
        json!({"voices": [], "pagination_token": null}),
    )]);
    let next_voice_service = service(&next_transport);
    next_voice_service
        .list_custom_voices(
            &XaiCustomVoiceListRequest::new().after(next),
            &credentials("xai-key"),
        )
        .await
        .unwrap();
    let sent = next_transport.sent.lock().unwrap();
    assert_eq!(sent[0].method, "GET");
    assert!(sent[0]
        .url
        .starts_with("http://127.0.0.1:8391/v1/custom-voices?"));
    assert!(sent[0].url.contains("pagination_token=cursor+%2F%2B%3D"));
}

#[tokio::test]
async fn scoped_get_update_and_delete_use_documented_routes_and_patch_null_semantics() {
    let transport = MockTransport::new([
        json_reply(200, Vec::new(), voice("nlbqfwie")),
        json_reply(
            200,
            Vec::new(),
            json!({"voice_id":"nlbqfwie","name":null,"tone":"calm"}),
        ),
        json_reply(200, Vec::new(), json!({"deleted":true})),
    ]);
    let voice_service = service(&transport);
    let existing = voice_service
        .get_custom_voice(
            &XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap(),
            &credentials("xai-key"),
        )
        .await
        .unwrap();
    let reference = existing.reference.clone();
    voice_service
        .update_custom_voice(
            &reference,
            &XaiCustomVoicePatch::new()
                .clear_name()
                .clear_description()
                .with_tone(XaiCustomVoiceTone::Calm),
            &credentials("xai-key"),
        )
        .await
        .unwrap();
    let deleted = voice_service
        .delete_custom_voice(&reference, &credentials("xai-key"))
        .await
        .unwrap();
    assert!(deleted.deleted);
    assert_eq!(deleted.reference, reference);

    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 3);
    assert_eq!(sent[0].method, "GET");
    assert_eq!(
        sent[0].url,
        "http://127.0.0.1:8391/v1/custom-voices/nlbqfwie"
    );
    assert_eq!(sent[1].method, "PATCH");
    assert_eq!(sent[1].url, sent[0].url);
    let body: Value = serde_json::from_slice(&sent[1].body).unwrap();
    assert_eq!(body, json!({"name":null,"description":null,"tone":"calm"}));
    assert_eq!(sent[2].method, "DELETE");
    assert_eq!(sent[2].url, sent[0].url);
}

#[tokio::test]
async fn get_custom_voice_accepts_a_null_cleared_name() {
    let transport = MockTransport::new([json_reply(
        200,
        Vec::new(),
        json!({"voice_id":"nlbqfwie","name":null}),
    )]);
    let voice_service = service(&transport);
    let reference = XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap();
    let result = voice_service
        .get_custom_voice(&reference, &credentials("xai-key"))
        .await
        .unwrap();
    assert_eq!(result.name, None);
    assert_eq!(result.reference, reference);
}

#[tokio::test]
async fn list_rejects_duplicate_ids_and_a_cursor_that_does_not_advance() {
    let duplicate_transport = MockTransport::new([json_reply(
        200,
        Vec::new(),
        json!({
            "voices": [voice("nlbqfwie"), voice("nlbqfwie")],
            "pagination_token": null
        }),
    )]);
    let duplicate_service = service(&duplicate_transport);
    let duplicate_error = duplicate_service
        .list_custom_voices(&XaiCustomVoiceListRequest::new(), &credentials("xai-key"))
        .await
        .unwrap_err();
    assert!(matches!(
        duplicate_error,
        XaiAudioError::InvalidResponse {
            operation: "list-custom-voices",
            ..
        }
    ));

    let cursor_transport = MockTransport::new([json_reply(
        200,
        Vec::new(),
        json!({"voices": [], "pagination_token":"cursor-1"}),
    )]);
    let cursor_service = service(&cursor_transport);
    let first = cursor_service
        .list_custom_voices(&XaiCustomVoiceListRequest::new(), &credentials("xai-key"))
        .await
        .unwrap();
    let cursor = first.next_page.unwrap();
    assert!(!format!("{cursor:?}").contains("cursor-1"));

    let stalled_transport = MockTransport::new([json_reply(
        200,
        Vec::new(),
        json!({"voices": [], "pagination_token":"cursor-1"}),
    )]);
    let stalled_service = service(&stalled_transport);
    let error = stalled_service
        .list_custom_voices(
            &XaiCustomVoiceListRequest::new().after(cursor),
            &credentials("xai-key"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        XaiAudioError::InvalidResponse {
            operation: "list-custom-voices",
            ..
        }
    ));
}

#[tokio::test]
async fn custom_voice_audio_is_streamed_with_actual_content_type_and_caller_byte_bound() {
    let transport = MockTransport::new([audio_reply(
        "audio/wav; rate=24000",
        [b"abc".as_slice(), b"de".as_slice()],
    )]);
    let voice_service = service(&transport);
    let reference = XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap();
    let mut audio = voice_service
        .get_custom_voice_audio(&reference, 4, &credentials("xai-key"))
        .await
        .unwrap();
    assert_eq!(audio.content_type, "audio/wav; rate=24000");
    assert_eq!(audio.reference, reference);
    assert_eq!(
        audio.next_chunk().await.unwrap().unwrap(),
        Bytes::from_static(b"abc")
    );
    let error = audio.next_chunk().await.unwrap_err();
    assert_eq!(error.delivered_bytes, 3);
    assert!(format!("{error}").contains("caller's byte limit"));
    assert!(audio.next_chunk().await.unwrap().is_none());

    let success_transport = MockTransport::new([audio_reply(
        "audio/mpeg",
        [b"a".as_slice(), b"b".as_slice()],
    )]);
    let success_service = service(&success_transport);
    let mut complete = success_service
        .get_custom_voice_audio(
            &XaiCustomVoiceRef::new(success_service.scope().clone(), "nlbqfwie").unwrap(),
            2,
            &credentials("xai-key"),
        )
        .await
        .unwrap();
    let mut all = Vec::new();
    while let Some(chunk) = complete.next_chunk().await.unwrap() {
        all.extend_from_slice(&chunk);
    }
    assert_eq!(all, b"ab");
}

#[tokio::test]
async fn terminal_audio_stream_errors_drop_the_transport_body_while_wrapper_is_retained() {
    let oversized_transport = DropSentinelTransport::new(vec![
        Ok(Bytes::from_static(b"abc")),
        Ok(Bytes::from_static(b"de")),
    ]);
    let dropped = oversized_transport.dropped.clone();
    let voice_service = service(&oversized_transport);
    let reference = XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap();
    let mut audio = voice_service
        .get_custom_voice_audio(&reference, 3, &credentials("xai-key"))
        .await
        .unwrap();
    assert!(!dropped.load(Ordering::SeqCst));
    assert_eq!(
        audio.next_chunk().await.unwrap().unwrap(),
        Bytes::from_static(b"abc")
    );
    assert!(audio.next_chunk().await.is_err());
    assert!(dropped.load(Ordering::SeqCst));
    assert!(audio.next_chunk().await.unwrap().is_none());

    let interrupted_transport = DropSentinelTransport::new(vec![
        Ok(Bytes::from_static(b"a")),
        Err(LlmError::Transport {
            message: "test response failure".into(),
        }),
    ]);
    let dropped = interrupted_transport.dropped.clone();
    let voice_service = service(&interrupted_transport);
    let reference = XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap();
    let mut audio = voice_service
        .get_custom_voice_audio(&reference, 10, &credentials("xai-key"))
        .await
        .unwrap();
    assert_eq!(
        audio.next_chunk().await.unwrap().unwrap(),
        Bytes::from_static(b"a")
    );
    assert!(audio.next_chunk().await.is_err());
    assert!(dropped.load(Ordering::SeqCst));
    assert!(audio.next_chunk().await.unwrap().is_none());
}

#[tokio::test]
async fn preflight_scope_duration_pagination_and_empty_patch_errors_do_not_send() {
    let transport = MockTransport::new([]);
    let voice_service = service(&transport);
    let wrong_service = XaiAudioService::new(
        &transport,
        XaiAudioConfig::new("other", "team/other").with_api_base_url("http://127.0.0.1:8391/v1"),
    )
    .unwrap();
    let wrong_ref = XaiCustomVoiceRef::new(wrong_service.scope().clone(), "nlbqfwie").unwrap();
    assert!(matches!(
        voice_service
            .get_custom_voice(&wrong_ref, &credentials("xai-key"))
            .await,
        Err(XaiAudioError::InvalidRequest(_))
    ));
    for duration in [f64::NAN, 0.0, 120.1, f64::INFINITY] {
        let request = XaiCustomVoiceCreateRequest::new(AudioInput::from_bytes(
            "clip.wav",
            "audio/wav",
            Bytes::from_static(b"audio"),
        ))
        .with_duration_seconds(duration);
        assert!(matches!(
            voice_service
                .create_custom_voice(request, &credentials("xai-key"))
                .await,
            Err(XaiAudioError::InvalidRequest(_))
        ));
    }
    assert!(matches!(
        voice_service
            .list_custom_voices(
                &XaiCustomVoiceListRequest::new().with_limit(0),
                &credentials("xai-key")
            )
            .await,
        Err(XaiAudioError::InvalidRequest(_))
    ));
    assert!(matches!(
        voice_service
            .update_custom_voice(
                &XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap(),
                &XaiCustomVoicePatch::new().with_name(""),
                &credentials("xai-key")
            )
            .await,
        Err(XaiAudioError::InvalidRequest(_))
    ));
    assert!(matches!(
        voice_service
            .get_custom_voice_audio(
                &XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap(),
                0,
                &credentials("xai-key")
            )
            .await,
        Err(XaiAudioError::InvalidRequest(_))
    ));
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn write_dispatch_failures_are_outcome_unknown_and_never_retried() {
    let failed = || {
        Outcome::Transport(LlmError::TransportTimeout {
            message: "request timed out".into(),
        })
    };
    let create_transport = MockTransport::new([failed()]);
    let create_error = service(&create_transport)
        .create_custom_voice(
            XaiCustomVoiceCreateRequest::new(AudioInput::from_bytes(
                "clip.wav",
                "audio/wav",
                Bytes::from_static(b"audio"),
            )),
            &credentials("xai-key"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        create_error,
        XaiAudioError::OutcomeUnknown {
            operation: "create-custom-voice",
            response: None,
            ..
        }
    ));
    assert_eq!(create_transport.sent.lock().unwrap().len(), 1);

    for (method, operation, outcome) in [
        ("update", "update-custom-voice", failed()),
        ("delete", "delete-custom-voice", failed()),
    ] {
        let transport = MockTransport::new([outcome]);
        let service = service(&transport);
        let reference = XaiCustomVoiceRef::new(service.scope().clone(), "nlbqfwie").unwrap();
        let error = if method == "update" {
            service
                .update_custom_voice(
                    &reference,
                    &XaiCustomVoicePatch::new().with_tone(XaiCustomVoiceTone::Warm),
                    &credentials("xai-key"),
                )
                .await
                .unwrap_err()
        } else {
            service
                .delete_custom_voice(&reference, &credentials("xai-key"))
                .await
                .unwrap_err()
        };
        assert!(matches!(
            error,
            XaiAudioError::OutcomeUnknown {
                operation: actual,
                response: None,
                ..
            } if actual == operation
        ));
        assert_eq!(transport.sent.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn successful_mutations_with_invalid_acknowledgements_are_outcome_unknown() {
    let create_transport = MockTransport::new([json_reply(
        201,
        vec![("x-request-id".into(), "create-ack-uncertain".into())],
        json!({"voice_id":"short","name":"Broken"}),
    )]);
    let create_error = service(&create_transport)
        .create_custom_voice(
            XaiCustomVoiceCreateRequest::new(AudioInput::from_bytes(
                "clip.wav",
                "audio/wav",
                Bytes::from_static(b"audio"),
            )),
            &credentials("xai-key"),
        )
        .await
        .unwrap_err();
    match create_error {
        XaiAudioError::OutcomeUnknown {
            operation: "create-custom-voice",
            response: Some(response),
            ..
        } => {
            assert_eq!(response.request_id.as_deref(), Some("create-ack-uncertain"));
            assert_eq!(
                *response.native,
                json!({"voice_id":"short","name":"Broken"})
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }

    let update_transport = MockTransport::new([json_reply(
        200,
        vec![("request-id".into(), "update-ack-uncertain".into())],
        voice("evemusic"),
    )]);
    let update_service = service(&update_transport);
    let reference = XaiCustomVoiceRef::new(update_service.scope().clone(), "nlbqfwie").unwrap();
    let update_error = update_service
        .update_custom_voice(
            &reference,
            &XaiCustomVoicePatch::new().with_tone(XaiCustomVoiceTone::Calm),
            &credentials("xai-key"),
        )
        .await
        .unwrap_err();
    match update_error {
        XaiAudioError::OutcomeUnknown {
            operation: "update-custom-voice",
            response: Some(response),
            ..
        } => {
            assert_eq!(response.request_id.as_deref(), Some("update-ack-uncertain"));
            assert_eq!(response.native["voice_id"], "evemusic");
        }
        other => panic!("unexpected error: {other:?}"),
    }

    let delete_transport = MockTransport::new([json_reply(
        200,
        vec![("request-id".into(), "delete-ack-uncertain".into())],
        json!({"deleted":false}),
    )]);
    let delete_service = service(&delete_transport);
    let reference = XaiCustomVoiceRef::new(delete_service.scope().clone(), "nlbqfwie").unwrap();
    let delete_error = delete_service
        .delete_custom_voice(&reference, &credentials("xai-key"))
        .await
        .unwrap_err();
    match delete_error {
        XaiAudioError::OutcomeUnknown {
            operation: "delete-custom-voice",
            response: Some(response),
            ..
        } => {
            assert_eq!(response.request_id.as_deref(), Some("delete-ack-uncertain"));
            assert_eq!(*response.native, json!({"deleted":false}));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn malformed_responses_and_provider_errors_keep_operation_context() {
    let malformed = [
        json!({"voice_id":"short","name":"Broken"}),
        json!({"voice_id":"nlbqfwie","name":4}),
        voice("evemusic"),
    ];
    for value in malformed {
        let transport = MockTransport::new([json_reply(200, Vec::new(), value)]);
        let voice_service = service(&transport);
        let reference = XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap();
        let error = voice_service
            .get_custom_voice(&reference, &credentials("xai-key"))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            XaiAudioError::InvalidResponse {
                operation: "get-custom-voice",
                ..
            }
        ));
    }

    let transport = MockTransport::new([json_reply(
        403,
        vec![("x-request-id".into(), "not-enabled".into())],
        json!({"error":{"message":"custom voices not enabled"}}),
    )]);
    let voice_service = service(&transport);
    let reference = XaiCustomVoiceRef::new(voice_service.scope().clone(), "nlbqfwie").unwrap();
    let error = voice_service
        .get_custom_voice(&reference, &credentials("xai-key"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        XaiAudioError::Provider {
            operation: "get-custom-voice",
            status: 403,
            request_id: Some(ref id),
            ..
        } if id == "not-enabled"
    ));
}
