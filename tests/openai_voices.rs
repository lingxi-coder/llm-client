use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{protocol::*, providers::openai::audio::*, *};
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
    content_length: u64,
}

struct Mock {
    replies: Mutex<VecDeque<(u16, Vec<u8>)>>,
    sent: Mutex<Vec<Sent>>,
    fail_next_after_send: Mutex<bool>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.record(
            request.method,
            request.url,
            request.headers,
            request.body.to_vec(),
            request.body.len() as u64,
        )
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let mut body = Vec::new();
        let mut chunks = request.body;
        while let Some(chunk) = chunks.next().await {
            body.extend_from_slice(&chunk?);
        }
        self.record(
            request.method,
            request.url,
            request.headers,
            body,
            request.content_length,
        )
    }
}

impl Mock {
    fn record(
        &self,
        method: String,
        url: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        content_length: u64,
    ) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(Sent {
            method,
            url,
            headers,
            body,
            content_length,
        });
        if std::mem::take(&mut *self.fail_next_after_send.lock().unwrap()) {
            return Err(LlmError::TransportTimeout {
                message: "mock interrupted after upload".into(),
            });
        }
        let (status, body) = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected OpenAI voice request");
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "req-voice".into())],
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
    setup_profiles(replies, vec![profile()])
}

fn setup_profiles(
    replies: Vec<(u16, Vec<u8>)>,
    profiles: Vec<ProviderProfile>,
) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        sent: Mutex::new(vec![]),
        fail_next_after_send: Mutex::new(false),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("project-key".into())),
        account_scope: Some("openai-project".into()),
        ..Default::default()
    }
}

fn consent_json(id: &str) -> Value {
    json!({
        "id": id,
        "created_at": 1734220800,
        "language": "en-US",
        "name": "John Doe",
        "object": "audio.voice_consent"
    })
}

fn bytes(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}

fn audio(filename: &str, media_type: &str, bytes: &'static [u8]) -> AudioInput {
    AudioInput::from_bytes(filename, media_type, Bytes::from_static(bytes))
}

#[tokio::test]
async fn documented_consent_lifecycle_and_voice_creation_use_exact_routes_and_fields() {
    let replies = vec![
        (
            200,
            bytes(json!({"catalog": [{"language": "en", "phrase": "Read this."}]})),
        ),
        (200, bytes(consent_json("cons_1234"))),
        (
            200,
            bytes(json!({
                "object":"list",
                "data":[consent_json("cons_1234")],
                "first_id":"cons_1234",
                "last_id":"cons_1234",
                "has_more":false
            })),
        ),
        (200, bytes(consent_json("cons_1234"))),
        (
            200,
            bytes(json!({
                "id":"cons_1234",
                "created_at":1734220800,
                "language":"en-US",
                "name":"John updated",
                "object":"audio.voice_consent"
            })),
        ),
        (
            200,
            bytes(json!({
                "id":"voice_5678",
                "created_at":1734220801,
                "name":"John voice",
                "object":"audio.voice"
            })),
        ),
        (
            200,
            bytes(json!({
                "id":"cons_1234",
                "deleted":true,
                "object":"audio.voice_consent"
            })),
        ),
    ];
    let (client, mock) = setup(replies);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap();
    let service = service_provider.audio().voices();

    let phrases = service.list_consent_phrases(&options()).await.unwrap();
    assert_eq!(phrases["catalog"][0]["language"], "en");

    let created_consent = service
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "John Doe".into(),
                language: "en-US".into(),
            },
            audio("consent.wav", "audio/x-wav", b"consent bytes"),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(created_consent.id, "cons_1234");

    let page = service
        .list_consents(
            &VoiceConsentListQuery {
                after: Some("cons_previous".into()),
                limit: Some(20),
            },
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(page.data[0].id, "cons_1234");

    assert_eq!(
        service
            .get_consent(created_consent.reference(), &options())
            .await
            .unwrap()
            .language,
        "en-US"
    );
    assert_eq!(
        service
            .update_consent(
                created_consent.reference(),
                &VoiceConsentUpdateRequest {
                    name: "John updated".into(),
                },
                &options(),
            )
            .await
            .unwrap()
            .name,
        "John updated"
    );
    let voice = service
        .create_voice(
            &CustomVoiceCreateRequest {
                name: "John voice".into(),
                consent: created_consent.reference().clone(),
            },
            audio("sample.wav", "audio/wav", b"voice sample bytes"),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(voice.id, "voice_5678");
    assert_eq!(voice.reference().id(), "voice_5678");
    assert_eq!(voice.reference().provider_id(), "openai");
    assert_eq!(voice.reference().profile_name(), "openai");
    assert_eq!(voice.reference().account_scope(), "openai-project");
    assert_eq!(
        voice.reference().endpoint_fingerprint(),
        files::provider_file_endpoint_fingerprint("https://api.openai.com/v1/audio/speech")
    );
    assert_ne!(
        voice.reference().endpoint_fingerprint(),
        files::provider_file_endpoint_fingerprint("https://api.openai.com/v1/audio/voices")
    );
    let imported = service
        .import_approved_voice(voice.id.clone(), &options())
        .unwrap();
    assert_eq!(voice.reference(), &imported);
    let saved_reference = serde_json::to_value(voice.reference()).unwrap();
    assert_eq!(saved_reference["voice_id"], "voice_5678");
    assert_eq!(
        serde_json::from_value::<CustomVoiceRef>(saved_reference.clone()).unwrap(),
        imported
    );
    assert!(!saved_reference.to_string().contains("api.openai.com"));
    assert!(
        service
            .delete_consent(created_consent.reference(), &options())
            .await
            .unwrap()
            .deleted
    );

    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 7);
    assert_eq!(sent[0].method, "GET");
    assert_eq!(
        sent[0].url,
        "https://api.openai.com/v1/audio/consent_phrases"
    );
    assert_eq!(sent[1].method, "POST");
    assert_eq!(
        sent[1].url,
        "https://api.openai.com/v1/audio/voice_consents"
    );
    assert_eq!(
        sent[2].url,
        "https://api.openai.com/v1/audio/voice_consents?after=cons_previous&limit=20"
    );
    assert_eq!(
        sent[3].url,
        "https://api.openai.com/v1/audio/voice_consents/cons_1234"
    );
    assert_eq!(sent[4].method, "POST");
    assert_eq!(
        sent[4].url,
        "https://api.openai.com/v1/audio/voice_consents/cons_1234"
    );
    let update: Value = serde_json::from_slice(&sent[4].body).unwrap();
    assert_eq!(update, json!({"name":"John updated"}));
    assert_eq!(sent[5].url, "https://api.openai.com/v1/audio/voices");
    assert_eq!(sent[6].method, "DELETE");

    for request in [&sent[1], &sent[5]] {
        assert_eq!(request.content_length, request.body.len() as u64);
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| { name == "authorization" && value == "Bearer project-key" }));
        let content_type = request
            .headers
            .iter()
            .find(|(name, _)| name == "content-type")
            .map(|(_, value)| value.as_str())
            .unwrap();
        assert!(content_type.starts_with("multipart/form-data; boundary="));
        let body = String::from_utf8_lossy(&request.body);
        assert!(body.contains("name=\"name\""));
    }
    let consent_body = String::from_utf8_lossy(&sent[1].body);
    assert!(consent_body.contains("name=\"language\""));
    assert!(consent_body.contains("John Doe"));
    assert!(consent_body.contains("en-US"));
    assert!(consent_body.contains("name=\"recording\"; filename=\"consent.wav\""));
    assert!(consent_body.contains("Content-Type: audio/x-wav"));
    assert!(consent_body.contains("consent bytes"));

    let voice_body = String::from_utf8_lossy(&sent[5].body);
    assert!(voice_body.contains("name=\"consent\""));
    assert!(voice_body.contains("John voice"));
    assert!(voice_body.contains("cons_1234"));
    assert!(voice_body.contains("name=\"audio_sample\"; filename=\"sample.wav\""));
    assert!(voice_body.contains("Content-Type: audio/wav"));
    assert!(voice_body.contains("voice sample bytes"));
}

#[tokio::test]
async fn recording_validation_rejects_unsupported_mime_and_oversize_before_dispatch() {
    let (client, mock) = setup(vec![(200, bytes(consent_json("cons_1234")))]);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap();
    let service = service_provider.audio().voices();

    let error = service
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "speaker".into(),
                language: "en-US".into(),
            },
            audio("consent.wav", "audio/wav; codecs=1", b"sample"),
            &options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        VoiceResourceError::Llm(LlmError::InvalidRequest { .. })
    ));

    let consent = service
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "speaker".into(),
                language: "en-US".into(),
            },
            audio("consent.wav", "audio/wav", b"consent"),
            &options(),
        )
        .await
        .unwrap();

    let oversized = AudioInput {
        filename: "sample.wav".into(),
        media_type: "audio/wav".into(),
        size_bytes: 10 * 1024 * 1024 + 1,
        body: stream::empty().boxed(),
    };
    let error = service
        .create_voice(
            &CustomVoiceCreateRequest {
                name: "speaker".into(),
                consent: consent.reference().clone(),
            },
            oversized,
            &options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        VoiceResourceError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn consent_references_reject_account_and_profile_mismatches_before_http() {
    let mut alternate = profile();
    alternate.profile_name = "openai-alternate".into();
    let (client, mock) = setup_profiles(
        vec![(200, bytes(consent_json("cons_1234")))],
        vec![profile(), alternate],
    );
    let service_provider = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap();
    let service = service_provider.audio().voices();
    let consent = service
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "speaker".into(),
                language: "en-US".into(),
            },
            audio("consent.wav", "audio/wav", b"consent"),
            &options(),
        )
        .await
        .unwrap();

    let mut another_account = options();
    another_account.account_scope = Some("different-project".into());
    let account_error = service
        .get_consent(consent.reference(), &another_account)
        .await
        .unwrap_err();
    assert!(matches!(
        account_error,
        VoiceResourceError::Llm(LlmError::InvalidRequest { .. })
    ));

    let alternate_provider = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai-alternate")
        .unwrap();
    let profile_error = alternate_provider
        .audio()
        .voices()
        .get_consent(consent.reference(), &options())
        .await
        .unwrap_err();
    assert!(matches!(
        profile_error,
        VoiceResourceError::Llm(LlmError::InvalidRequest { .. })
    ));

    let mut wrong_endpoint_value = serde_json::to_value(consent.reference()).unwrap();
    wrong_endpoint_value["endpoint_fingerprint"] = Value::String("different-endpoint".into());
    let wrong_endpoint: VoiceConsentRef = serde_json::from_value(wrong_endpoint_value).unwrap();
    let endpoint_error = service
        .get_consent(&wrong_endpoint, &options())
        .await
        .unwrap_err();
    assert!(matches!(
        endpoint_error,
        VoiceResourceError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn existing_voice_import_requires_explicit_profile_and_account_scope() {
    let (client, mock) = setup(vec![]);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap();
    let service = service_provider.audio().voices();
    let mut missing_scope = options();
    missing_scope.account_scope = None;
    assert!(matches!(
        service.import_approved_voice("voice_1234", &missing_scope),
        Err(VoiceResourceError::Llm(LlmError::InvalidRequest { .. }))
    ));
    let mut padded_scope = options();
    padded_scope.account_scope = Some(" openai-project ".into());
    assert!(matches!(
        service.import_approved_voice("voice_1234", &padded_scope),
        Err(VoiceResourceError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(matches!(
        service.import_approved_voice("not-a-voice-id", &options()),
        Err(VoiceResourceError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(matches!(
        client.provider::<lingxi_llm_client::providers::OpenAiClient>("unknown-profile"),
        Err(lingxi_llm_client::providers::ProviderBindingError::UnknownProfile { .. })
    ));
    assert!(mock.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn created_voice_reference_is_accepted_by_speech_without_leaking_scope() {
    let (client, mock) = setup(vec![
        (200, bytes(consent_json("cons_1234"))),
        (
            200,
            bytes(json!({
                "id":"voice_5678",
                "created_at":1734220801,
                "name":"John voice",
                "object":"audio.voice"
            })),
        ),
        (200, b"audio bytes".to_vec()),
    ]);
    let voices_provider = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap();
    let voices = voices_provider.audio().voices();
    let consent = voices
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "speaker".into(),
                language: "en-US".into(),
            },
            audio("consent.wav", "audio/wav", b"consent"),
            &options(),
        )
        .await
        .unwrap();
    let voice = voices
        .create_voice(
            &CustomVoiceCreateRequest {
                name: "speaker voice".into(),
                consent: consent.reference().clone(),
            },
            audio("sample.wav", "audio/wav", b"sample"),
            &options(),
        )
        .await
        .unwrap();
    let request = SpeechRequest {
        model: SpeechModel::Gpt4oMiniTts,
        input: "hello".into(),
        voice: SpeechVoice::Custom(voice.reference().clone()),
        format: SpeechFormat::Mp3,
        instructions: None,
        speed: None,
    };
    let mut speech = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .audio()
        .synthesize(&request, &options())
        .await
        .unwrap();
    assert_eq!(
        speech.next_chunk().await.unwrap().unwrap().as_ref(),
        b"audio bytes"
    );
    assert!(speech.next_chunk().await.unwrap().is_none());

    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 3);
    let speech_body: Value = serde_json::from_slice(&sent[2].body).unwrap();
    assert_eq!(speech_body["voice"], json!({"id":"voice_5678"}));
    assert_eq!(speech_body["voice"].as_object().unwrap().len(), 1);
}

#[tokio::test]
async fn malformed_success_after_voice_creation_reports_unknown_outcome() {
    let (client, mock) = setup(vec![
        (200, bytes(consent_json("cons_1234"))),
        (
            200,
            bytes(json!({
                "id":"voice_5678",
                "created_at":1734220801,
                "name":"speaker voice",
                "object":"audio.voice_consent"
            })),
        ),
    ]);
    let voices_provider = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap();
    let voices = voices_provider.audio().voices();
    let consent = voices
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "speaker".into(),
                language: "en-US".into(),
            },
            audio("consent.wav", "audio/wav", b"consent"),
            &options(),
        )
        .await
        .unwrap();
    let error = voices
        .create_voice(
            &CustomVoiceCreateRequest {
                name: "speaker voice".into(),
                consent: consent.reference().clone(),
            },
            audio("sample.wav", "audio/wav", b"sample"),
            &options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        VoiceResourceError::OutcomeUnknown {
            operation: "creating a custom voice",
            ..
        }
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn interrupted_consent_upload_reports_unknown_outcome_without_retry() {
    let (client, mock) = setup(vec![]);
    *mock.fail_next_after_send.lock().unwrap() = true;
    let result = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .audio()
        .voices()
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "speaker".into(),
                language: "en-US".into(),
            },
            audio("consent.wav", "audio/wav", b"consent"),
            &options(),
        )
        .await;
    assert!(matches!(
        result,
        Err(VoiceResourceError::OutcomeUnknown {
            operation: "creating a voice consent",
            ..
        })
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}
