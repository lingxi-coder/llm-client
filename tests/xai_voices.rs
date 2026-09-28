use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::xai::audio::{XaiAudioConfig, XaiAudioCredentials, XaiAudioError, XaiAudioService},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex, time::Duration};

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[derive(Debug)]
struct SentRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Bytes,
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

    fn response(reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: stream::iter([Ok(Bytes::from(reply.body))]).boxed(),
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
            body: request.body,
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("a voice-list response is queued");
        Ok(Self::response(reply))
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "voice listing uses a regular GET request".into(),
        })
    }
}

fn reply(status: u16, headers: Vec<(String, String)>, body: Value) -> Reply {
    Reply {
        status,
        headers,
        body: serde_json::to_vec(&body).unwrap(),
    }
}

fn credentials(key: &str) -> XaiAudioCredentials {
    XaiAudioCredentials::new(Secret::new(key.to_owned()))
}

fn make_service(transport: &MockTransport) -> XaiAudioService<'_> {
    XaiAudioService::new(
        transport,
        XaiAudioConfig::new("work", "team/account-7")
            .with_api_base_url("http://127.0.0.1:8391/v1")
            .with_request_timeout(Duration::from_secs(2)),
    )
    .unwrap()
}

#[tokio::test]
async fn list_voices_uses_documented_get_and_preserves_native_metadata() {
    let native = json!({
        "voices": [
            {
                "voice_id": "eve",
                "name": "Eve",
                "language": "multilingual",
                "accent": "British"
            },
            {"voice_id": "ara", "name": "Ara", "provider_extension": {"tone": "warm"}}
        ],
        "provider_extension": "kept"
    });
    let transport = MockTransport::new([reply(
        200,
        vec![("x-request-id".into(), "voice-list-1".into())],
        native.clone(),
    )]);
    let service = make_service(&transport);
    let voices = service
        .list_voices(&credentials("xai-test-key"))
        .await
        .unwrap();

    assert_eq!(voices.scope.account_scope, "team/account-7");
    assert_eq!(voices.request_id.as_deref(), Some("voice-list-1"));
    assert_eq!(voices.voices.len(), 2);
    assert_eq!(voices.voices[0].voice_id, "eve");
    assert_eq!(voices.voices[0].name, "Eve");
    assert_eq!(voices.voices[0].language.as_deref(), Some("multilingual"));
    assert_eq!(voices.voices[0].native, native["voices"][0]);
    assert_eq!(voices.voices[1].language, None);
    assert_eq!(voices.voices[1].native, native["voices"][1]);
    assert_eq!(voices.native, native);

    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "GET");
    assert_eq!(sent[0].url, "http://127.0.0.1:8391/v1/tts/voices");
    assert!(sent[0].body.is_empty());
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer xai-test-key"
    }));
}

#[tokio::test]
async fn list_voices_reports_provider_errors_with_request_id_and_body() {
    let transport = MockTransport::new([reply(
        403,
        vec![("request-id".into(), "voice-list-denied".into())],
        json!({"error":{"message":"forbidden"}}),
    )]);
    let error = make_service(&transport)
        .list_voices(&credentials("xai-test-key"))
        .await
        .unwrap_err();

    match error {
        XaiAudioError::Provider {
            operation,
            status,
            request_id,
            body,
        } => {
            assert_eq!(operation, "list-voices");
            assert_eq!(status, 403);
            assert_eq!(request_id.as_deref(), Some("voice-list-denied"));
            assert_eq!(*body, json!({"error":{"message":"forbidden"}}));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn list_voices_rejects_malformed_top_level_empty_and_duplicate_ids() {
    let malformed = [
        json!({"items": []}),
        json!({"voices": {}}),
        json!({"voices": [{"voice_id": "  ", "name": "Eve"}]}),
        json!({
            "voices": [
                {"voice_id": "Eve", "name": "Eve"},
                {"voice_id": "eve", "name": "Eve alias"}
            ]
        }),
    ];
    for native in malformed {
        let transport = MockTransport::new([reply(200, Vec::new(), native)]);
        let error = make_service(&transport)
            .list_voices(&credentials("xai-test-key"))
            .await
            .unwrap_err();
        assert!(
            matches!(
                &error,
                XaiAudioError::InvalidResponse {
                    operation: "list-voices",
                    ..
                }
            ),
            "unexpected error: {error:?}"
        );
    }
}

#[tokio::test]
async fn list_voices_validates_credentials_before_sending() {
    let transport = MockTransport::new([]);
    let error = make_service(&transport)
        .list_voices(&credentials("\n"))
        .await
        .unwrap_err();

    assert!(matches!(error, XaiAudioError::InvalidRequest(_)));
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn get_voice_uses_safe_path_segment_and_preserves_scoped_native_details() {
    let native = json!({
        "voice_id": "eve",
        "name": "Eve",
        "language": "multilingual",
        "accent": "British",
        "provider_extension": {"pitch": "warm"}
    });
    let transport = MockTransport::new([reply(
        200,
        vec![("x-request-id".into(), "voice-get-1".into())],
        native.clone(),
    )]);
    let service = make_service(&transport);
    let details = service
        .get_voice("eve", &credentials("xai-test-key"))
        .await
        .unwrap();

    assert_eq!(details.scope.account_scope, "team/account-7");
    assert_eq!(details.scope.profile_name, "work");
    assert_eq!(details.voice.voice_id, "eve");
    assert_eq!(details.voice.name, "Eve");
    assert_eq!(details.voice.language.as_deref(), Some("multilingual"));
    assert_eq!(details.voice.native, native);
    assert_eq!(details.native, native);
    assert_eq!(details.request_id.as_deref(), Some("voice-get-1"));

    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "GET");
    assert_eq!(sent[0].url, "http://127.0.0.1:8391/v1/tts/voices/eve");
    assert!(sent[0].body.is_empty());
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer xai-test-key"
    }));
}

#[tokio::test]
async fn get_voice_requires_exact_response_id_and_valid_details() {
    let mismatched_native = json!({"voice_id":"Eve", "name":"Eve"});
    let transport = MockTransport::new([reply(
        200,
        vec![("request-id".into(), "voice-get-mismatch".into())],
        mismatched_native.clone(),
    )]);
    let error = make_service(&transport)
        .get_voice("eve", &credentials("xai-test-key"))
        .await
        .unwrap_err();
    match error {
        XaiAudioError::InvalidResponse {
            operation,
            request_id,
            native,
            ..
        } => {
            assert_eq!(operation, "get-voice");
            assert_eq!(request_id.as_deref(), Some("voice-get-mismatch"));
            assert_eq!(*native, mismatched_native);
        }
        other => panic!("unexpected mismatched-ID error: {other:?}"),
    }
    assert_eq!(transport.sent.lock().unwrap().len(), 1);

    for native in [
        json!({"voice_id":"eve"}),
        json!({"voice_id":"eve", "name":"  "}),
        json!({"voice_id":"eve", "name":"Eve", "language":false}),
    ] {
        let transport = MockTransport::new([reply(200, Vec::new(), native)]);
        let error = make_service(&transport)
            .get_voice("eve", &credentials("xai-test-key"))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            XaiAudioError::InvalidResponse {
                operation: "get-voice",
                ..
            }
        ));
    }
}

#[tokio::test]
async fn get_voice_rejects_unsafe_path_ids_and_invalid_credentials_before_http() {
    let transport = MockTransport::new([]);
    let service = make_service(&transport);
    for voice_id in [
        "",
        " ",
        " eve",
        "eve ",
        ".",
        "..",
        "eve/../ara",
        "eve\\ara",
        "eve%2e%2e",
        "eve%2fara",
        "eve%252fara",
        "eve\n",
    ] {
        assert!(matches!(
            service
                .get_voice(voice_id, &credentials("xai-test-key"))
                .await,
            Err(XaiAudioError::InvalidRequest(_))
        ));
    }
    assert!(matches!(
        service.get_voice("eve", &credentials("\n")).await,
        Err(XaiAudioError::InvalidRequest(_))
    ));
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn get_voice_preserves_provider_errors_without_retry() {
    let transport = MockTransport::new([reply(
        404,
        vec![("x-request-id".into(), "voice-get-missing".into())],
        json!({"error":{"message":"voice not found"}}),
    )]);
    let error = make_service(&transport)
        .get_voice("missing", &credentials("xai-test-key"))
        .await
        .unwrap_err();
    match error {
        XaiAudioError::Provider {
            operation,
            status,
            request_id,
            body,
        } => {
            assert_eq!(operation, "get-voice");
            assert_eq!(status, 404);
            assert_eq!(request_id.as_deref(), Some("voice-get-missing"));
            assert_eq!(*body, json!({"error":{"message":"voice not found"}}));
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}
