use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::google::speech::{
        GeminiSpeechModel, GeminiSpeechScope, GeminiVoiceAudioData, GeminiVoiceCreateRequest,
        GeminiVoiceListOptions, GeminiVoicesDispatch, GeminiVoicesScope, GeminiVoicesService,
        GEMINI_SPEECH_ENDPOINT, GEMINI_VOICES_ENDPOINT,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

enum Outcome {
    Response {
        status: u16,
        headers: Vec<(String, String)>,
        body: Bytes,
    },
    Transport(LlmError),
}

struct Mock {
    outcomes: Mutex<VecDeque<Outcome>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(outcomes: impl IntoIterator<Item = Outcome>) -> Self {
        Self {
            outcomes: Mutex::new(outcomes.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        match self
            .outcomes
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected Gemini Voices request")
        {
            Outcome::Transport(error) => Err(error),
            Outcome::Response {
                status,
                headers,
                body,
            } => Ok(StreamResponse {
                status,
                headers,
                body: stream::once(async move { Ok(body) }).boxed(),
            }),
        }
    }
}

fn response(status: u16, body: Value) -> Outcome {
    Outcome::Response {
        status,
        headers: vec![
            ("content-type".into(), "application/json".into()),
            ("x-goog-request-id".into(), "voices-request-1".into()),
        ],
        body: Bytes::from(serde_json::to_vec(&body).unwrap()),
    }
}

fn empty_response(status: u16) -> Outcome {
    Outcome::Response {
        status,
        headers: vec![("x-goog-request-id".into(), "voices-request-1".into())],
        body: Bytes::new(),
    }
}

fn service<'a>(mock: &'a Mock, account: &str) -> GeminiVoicesService<'a> {
    GeminiVoicesService::new(
        mock,
        GeminiVoicesScope::new("voices-profile", account, GEMINI_VOICES_ENDPOINT).unwrap(),
    )
    .unwrap()
}

fn credential() -> Secret<String> {
    Secret::new("google-test-key".to_owned())
}

fn voice_response(id: &str, voice_type: &str) -> Value {
    json!({
        "id": id,
        "type": voice_type,
        "display_name": "A test voice",
        "description": "Mocked provider metadata",
        "future_field": {"preserved": true}
    })
}

fn request_body(request: &HttpRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[tokio::test]
async fn create_prompted_sends_explicit_storage_and_returns_scoped_reference() {
    let mock = Mock::new([response(
        200,
        voice_response("voice_custom_123", "prompted"),
    )]);
    let service = service(&mock, "google-project-1");
    let created = service
        .create(
            &credential(),
            &GeminiVoiceCreateRequest::prompted("A warm narrator.")
                .with_display_name("Narrator")
                .with_model("gemini-3.8-flash-tts"),
        )
        .await
        .unwrap();

    let reference = created.reference.as_ref().unwrap();
    assert_eq!(reference.voice_id(), "voice_custom_123");
    assert_eq!(reference.profile_name(), "voices-profile");
    assert_eq!(reference.account_scope(), "google-project-1");
    assert_eq!(created.request_id.as_deref(), Some("voices-request-1"));
    assert_eq!(created.native["future_field"]["preserved"], true);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, GEMINI_VOICES_ENDPOINT);
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-goog-api-key") && value == "google-test-key"
    }));
    let body = request_body(&requests[0]);
    assert_eq!(body["store"], true);
    assert_eq!(body["voice"]["type"], "prompted");
    assert_eq!(body["voice"]["prompted"]["input"], "A warm narrator.");
    assert_eq!(body["voice"]["display_name"], "Narrator");
    assert_eq!(body["voice"]["model"], "gemini-3.8-flash-tts");
}

#[tokio::test]
async fn create_stateless_replication_encodes_consent_and_key_is_usable_redacted() {
    let key = "voicekey_ciphertext+/=";
    let mock = Mock::new([response(
        200,
        json!({"type":"replicated","key":key,"future_field":1}),
    )]);
    let service = service(&mock, "google-project-1");
    let created = service
        .create(
            &credential(),
            &GeminiVoiceCreateRequest::replicated(
                GeminiVoiceAudioData::new(Bytes::from_static(b"reference"), "audio/wav"),
                GeminiVoiceAudioData::new(Bytes::from_static(b"consent"), "audio/wav"),
                false,
            ),
        )
        .await
        .unwrap();

    assert!(created.reference.is_none());
    assert_eq!(created.synthesis_voice(), Some(key));
    let speech_scope =
        GeminiSpeechScope::new("voices-profile", "google-project-1", GEMINI_SPEECH_ENDPOINT)
            .unwrap();
    let speech = created
        .speech_request(
            &speech_scope,
            GeminiSpeechModel::Gemini38FlashTts,
            "Test key in speech request.",
        )
        .unwrap();
    let speech_json = serde_json::to_value(&speech).unwrap();
    assert_eq!(speech_json["voice"], key);
    assert!(!format!("{created:?}").contains(key));
    assert!(!format!("{speech:?}").contains(key));

    let requests = mock.requests.lock().unwrap();
    let body = request_body(&requests[0]);
    assert_eq!(body["store"], false);
    assert_eq!(body["voice"]["type"], "replicated");
    assert_eq!(
        body["voice"]["replicated"]["source_audio"]["data"],
        BASE64.encode(b"reference")
    );
    assert_eq!(
        body["voice"]["replicated"]["consent_audio"]["data"],
        BASE64.encode(b"consent")
    );
    assert_eq!(
        body["voice"]["replicated"]["source_audio"]["mime_type"],
        "audio/wav"
    );
    assert_eq!(
        body["voice"]["replicated"]["consent_audio"]["mime_type"],
        "audio/wav"
    );
}

#[tokio::test]
async fn create_rejects_mutually_present_id_and_key_as_accepted_malformed_response() {
    let returned_key = "voicekey_must_not_appear_in_debug";
    let mock = Mock::new([
        response(
            200,
            json!({"type":"prompted","id":"voice_abc","key":returned_key}),
        ),
        response(
            200,
            json!({"type":"replicated","id":"voice_abc","key":"voicekey_abc"}),
        ),
    ]);
    let service = service(&mock, "google-project-1");

    let stored = service
        .create(&credential(), &GeminiVoiceCreateRequest::prompted("prompt"))
        .await
        .unwrap_err();
    assert_eq!(stored.dispatch(), GeminiVoicesDispatch::Accepted);
    let debug = format!("{stored:?}");
    assert!(!debug.contains(returned_key));
    assert!(debug.contains("<provider response omitted from Debug>"));

    let stateless = service
        .create(
            &credential(),
            &GeminiVoiceCreateRequest::replicated(
                GeminiVoiceAudioData::new(Bytes::from_static(b"ref"), "audio/wav"),
                GeminiVoiceAudioData::new(Bytes::from_static(b"yes"), "audio/wav"),
                false,
            ),
        )
        .await
        .unwrap_err();
    assert_eq!(stateless.dispatch(), GeminiVoicesDispatch::Accepted);
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn list_repeats_filters_preserves_page_token_and_scopes_each_voice() {
    let mock = Mock::new([response(
        200,
        json!({
            "voices":[voice_response("voice_listed_1", "replicated"), voice_response("Puck", "prebuilt")],
            "next_page_token":"next/token+opaque",
            "private_native_sentinel":"not displayed in Debug"
        }),
    )]);
    let service = service(&mock, "google-project-1");
    let page = service
        .list(
            &credential(),
            &GeminiVoiceListOptions::new()
                .with_page_size(100)
                .with_page_token("prev/token+opaque")
                .with_accents(["American", "British"])
                .with_language_codes(["en-US"])
                .with_types([
                    lingxi_llm_client::providers::google::speech::GeminiVoiceType::Replicated,
                    lingxi_llm_client::providers::google::speech::GeminiVoiceType::Prebuilt,
                ])
                .with_search("warm voice"),
        )
        .await
        .unwrap();

    assert_eq!(page.voices.len(), 2);
    assert_eq!(
        page.native["private_native_sentinel"],
        "not displayed in Debug"
    );
    assert!(!format!("{page:?}").contains("not displayed in Debug"));
    assert_eq!(
        page.voices[0].reference.as_ref().unwrap().account_scope(),
        "google-project-1"
    );
    assert!(page.voices[1].reference.is_none());
    assert_eq!(page.next_page_token.as_deref(), Some("next/token+opaque"));
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[0].body, Bytes::new());
    let url = url::Url::parse(&requests[0].url).unwrap();
    let pairs: Vec<_> = url
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    assert!(pairs.contains(&("accent".into(), "American".into())));
    assert!(pairs.contains(&("accent".into(), "British".into())));
    assert!(pairs.contains(&("language_code".into(), "en-US".into())));
    assert!(pairs.contains(&("type".into(), "replicated".into())));
    assert!(pairs.contains(&("type".into(), "prebuilt".into())));
    assert!(pairs.contains(&("page_token".into(), "prev/token+opaque".into())));
    assert!(pairs.contains(&("search".into(), "warm voice".into())));
}

#[tokio::test]
async fn foreign_reference_and_unsafe_ids_are_rejected_before_dispatch() {
    let mock = Mock::new(std::iter::empty::<Outcome>());
    let service = service(&mock, "google-project-1");
    let foreign_scope =
        GeminiVoicesScope::new("voices-profile", "google-project-2", GEMINI_VOICES_ENDPOINT)
            .unwrap();
    let foreign = foreign_scope.voice_ref("voice_foreign").unwrap();
    let other_profile = GeminiVoicesScope::new(
        "another-profile",
        "google-project-1",
        GEMINI_VOICES_ENDPOINT,
    )
    .unwrap()
    .voice_ref("voice_other_profile")
    .unwrap();

    let error = service.delete(&credential(), &foreign).await.unwrap_err();
    assert_eq!(error.dispatch(), GeminiVoicesDispatch::NotSent);
    let error = service
        .get(&credential(), &other_profile)
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), GeminiVoicesDispatch::NotSent);
    assert!(
        GeminiVoicesScope::new("voices-profile", "google-project-1", GEMINI_VOICES_ENDPOINT,)
            .unwrap()
            .voice_ref("voice_../traversal")
            .is_err()
    );
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn mutation_transport_and_server_failures_are_unknown_without_retry() {
    let mock = Mock::new([
        Outcome::Transport(LlmError::TransportTimeout {
            message: "request outcome unknown".into(),
        }),
        response(408, json!({"error":{"message":"request timeout"}})),
        response(503, json!({"error":{"message":"temporary failure"}})),
        response(400, json!({"error":{"message":"invalid request"}})),
    ]);
    let service = service(&mock, "google-project-1");
    let request = GeminiVoiceCreateRequest::prompted("A test narrator.");

    let transport = service.create(&credential(), &request).await.unwrap_err();
    assert_eq!(transport.dispatch(), GeminiVoicesDispatch::Unknown);
    let timeout = service.create(&credential(), &request).await.unwrap_err();
    assert_eq!(timeout.dispatch(), GeminiVoicesDispatch::Unknown);
    let server = service.create(&credential(), &request).await.unwrap_err();
    assert_eq!(server.dispatch(), GeminiVoicesDispatch::Unknown);
    let rejection = service.create(&credential(), &request).await.unwrap_err();
    assert_eq!(rejection.dispatch(), GeminiVoicesDispatch::Rejected);
    assert_eq!(mock.requests.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn delete_transport_and_server_failures_are_unknown_without_retry() {
    let mock = Mock::new([
        Outcome::Transport(LlmError::TransportTimeout {
            message: "delete outcome unknown".into(),
        }),
        response(500, json!({"error":{"message":"temporary failure"}})),
        response(403, json!({"error":{"message":"not authorized"}})),
    ]);
    let service = service(&mock, "google-project-1");
    let reference = service.scope().voice_ref("voice_to_delete").unwrap();

    let transport = service.delete(&credential(), &reference).await.unwrap_err();
    assert_eq!(transport.dispatch(), GeminiVoicesDispatch::Unknown);
    let server = service.delete(&credential(), &reference).await.unwrap_err();
    assert_eq!(server.dispatch(), GeminiVoicesDispatch::Unknown);
    let rejection = service.delete(&credential(), &reference).await.unwrap_err();
    assert_eq!(rejection.dispatch(), GeminiVoicesDispatch::Rejected);
    assert_eq!(mock.requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn get_and_delete_use_only_scoped_stored_voice_resources() {
    let mock = Mock::new([
        response(200, voice_response("voice_abc123", "replicated")),
        empty_response(204),
    ]);
    let service = service(&mock, "google-project-1");
    let reference = service.scope().voice_ref("voice_abc123").unwrap();
    let fetched = service.get(&credential(), &reference).await.unwrap();
    assert_eq!(fetched.reference.as_ref(), Some(&reference));
    service.delete(&credential(), &reference).await.unwrap();

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(
        requests[0].url,
        format!("{GEMINI_VOICES_ENDPOINT}/voice_abc123")
    );
    assert_eq!(requests[1].method, "DELETE");
    assert_eq!(
        requests[1].url,
        format!("{GEMINI_VOICES_ENDPOINT}/voice_abc123")
    );
}
