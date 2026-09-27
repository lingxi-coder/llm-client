use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    files::ProviderFileRef,
    minimax_tts::MiniMaxTtsModel,
    minimax_voices::{
        MiniMaxVoiceClonePrompt, MiniMaxVoiceCloneRequest, MiniMaxVoiceDesignRequest,
        MiniMaxVoiceKind, MiniMaxVoiceLanguageBoost, MiniMaxVoiceListRequest, MiniMaxVoiceListType,
        MiniMaxVoiceRef, MiniMaxVoicesConfig, MiniMaxVoicesCredentials, MiniMaxVoicesDispatch,
        MiniMaxVoicesError, MiniMaxVoicesRegion, MiniMaxVoicesService,
    },
    protocol::{LlmError, ProtocolFamily, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex, time::Duration};

const API_ROOT: &str = "http://127.0.0.1:8420/v1";

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

enum Outcome {
    Response(Reply),
    Transport(LlmError),
}

struct MockTransport {
    outcomes: Mutex<VecDeque<Outcome>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    fn new(outcomes: impl IntoIterator<Item = Outcome>) -> Self {
        Self {
            outcomes: Mutex::new(outcomes.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        match self
            .outcomes
            .lock()
            .unwrap()
            .pop_front()
            .expect("an HTTP result is queued")
        {
            Outcome::Response(reply) => Ok(StreamResponse {
                status: reply.status,
                headers: reply.headers,
                body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
            }),
            Outcome::Transport(error) => Err(error),
        }
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "MiniMax voice management sends bounded JSON bodies".into(),
        })
    }
}

fn json_response(status: u16, request_id: &str, value: Value) -> Outcome {
    Outcome::Response(Reply {
        status,
        headers: vec![("x-request-id".into(), request_id.into())],
        body: serde_json::to_vec(&value).unwrap(),
    })
}

fn service(transport: &MockTransport, region: MiniMaxVoicesRegion) -> MiniMaxVoicesService<'_> {
    MiniMaxVoicesService::new(
        transport,
        MiniMaxVoicesConfig::new("voices-profile", "team/account-3", region)
            .with_api_base_url(API_ROOT)
            .with_request_timeout(Duration::from_secs(3)),
    )
    .unwrap()
}

fn credentials() -> MiniMaxVoicesCredentials {
    MiniMaxVoicesCredentials::new(Secret::new("minimax-test-key".to_owned()))
}

fn file_ref(service: &MiniMaxVoicesService<'_>, file_id: &str, purpose: &str) -> ProviderFileRef {
    ProviderFileRef {
        provider_id: service.scope().provider_id.clone(),
        profile_name: service.scope().profile_name.clone(),
        endpoint_fingerprint: service.scope().endpoint_fingerprint.clone(),
        account_scope: Some(service.scope().account_scope.clone()),
        protocol: ProtocolFamily::OpenAiChat,
        file_id: file_id.into(),
        uri: None,
        filename: Some("voice.wav".into()),
        media_type: Some("application/octet-stream".into()),
        size_bytes: Some(1_000_000),
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: Some(purpose.into()),
    }
}

fn base_resp() -> Value {
    json!({"status_code":0,"status_msg":"success"})
}

#[tokio::test]
async fn clone_uses_scoped_file_ids_preserves_native_response_and_does_not_add_preview() {
    // This mirrors the current successful clone example: no response voice_id
    // is documented, so the requested ID binds the returned reference.
    let native = json!({
        "input_sensitive": false,
        "input_sensitive_type": 0,
        "demo_audio": "",
        "extra_info": {
            "audio_length": 11124,
            "audio_sample_rate": 32000,
            "audio_size": 179926,
            "bitrate": 128000,
            "word_count": 18,
            "usage_characters": 18
        },
        "base_resp": base_resp(),
        "provider_extension": {"preserved": true}
    });
    let transport = MockTransport::new([json_response(200, "clone-req-1", native.clone())]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let request = MiniMaxVoiceCloneRequest::new(
        file_ref(&voice_service, "9001001", "voice_clone"),
        "Narrator_01",
    )
    .with_voice_audio_duration_seconds(90.0)
    .with_language_boost(MiniMaxVoiceLanguageBoost::ChineseYue)
    .with_text_validation("hello words")
    .with_accuracy(0.8)
    .with_noise_reduction(false)
    .with_volume_normalization(true)
    .with_aigc_watermark(false);

    let created = voice_service
        .clone_voice(&request, &credentials())
        .await
        .unwrap();
    assert_eq!(created.reference.voice_id(), "Narrator_01");
    assert_eq!(created.reference.kind(), MiniMaxVoiceKind::Cloned);
    assert_eq!(
        created.reference.scope().region,
        MiniMaxVoicesRegion::International
    );
    assert_eq!(created.input_sensitive, Some(json!(false)));
    assert_eq!(created.input_sensitive_type, Some(json!(0)));
    assert_eq!(created.demo_audio.as_deref(), Some(""));
    assert_eq!(
        created.extra_info.as_ref().unwrap().usage_characters,
        Some(18)
    );
    assert_eq!(created.native, native);
    assert_eq!(created.request_id.as_deref(), Some("clone-req-1"));

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, format!("{API_ROOT}/voice_clone"));
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer minimax-test-key"
    }));
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-type") && value == "application/json"
    }));
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["file_id"], 9_001_001_i64);
    assert_eq!(body["voice_id"], "Narrator_01");
    assert_eq!(body["language_boost"], "Chinese,Yue");
    assert_eq!(body["text_validation"], "hello words");
    assert_eq!(body["accuracy"], 0.8);
    assert_eq!(body["need_noise_reduction"], false);
    assert_eq!(body["need_volume_normalization"], true);
    assert_eq!(body["aigc_watermark"], false);
    assert!(body.get("text").is_none());
    assert!(body.get("model").is_none());
}

#[tokio::test]
async fn clone_prompt_and_paid_preview_are_sent_only_when_explicitly_requested() {
    let prompt_transport = MockTransport::new([json_response(
        200,
        "clone-req-2",
        json!({"base_resp":base_resp(),"input_sensitive":{},"input_sensitive_type":0}),
    )]);
    let voice_service = MiniMaxVoicesService::new(
        &prompt_transport,
        MiniMaxVoicesConfig::new(
            "voices-profile",
            "team/account-3",
            MiniMaxVoicesRegion::ChinaMainland,
        ),
    )
    .unwrap();
    let request = MiniMaxVoiceCloneRequest::new(
        file_ref(&voice_service, "81001", "voice_clone"),
        "ChinaVoice01",
    )
    .with_clone_prompt(
        MiniMaxVoiceClonePrompt::new(
            file_ref(&voice_service, "81002", "prompt_audio"),
            "prompt\ntranscript",
        )
        .with_duration_seconds(3.5),
    )
    .with_preview(
        "Hello,\nthis is a paid preview.",
        MiniMaxTtsModel::Speech28Hd,
    );
    let created = voice_service
        .clone_voice(&request, &credentials())
        .await
        .unwrap();
    assert_eq!(
        created.reference.scope().region,
        MiniMaxVoicesRegion::ChinaMainland
    );
    let body: Value =
        serde_json::from_slice(&prompt_transport.requests.lock().unwrap()[0].body).unwrap();
    assert_eq!(
        body["clone_prompt"],
        json!({"prompt_audio":81002,"prompt_text":"prompt\ntranscript"})
    );
    assert_eq!(body["text"], "Hello,\nthis is a paid preview.");
    assert_eq!(body["model"], "speech-2.8-hd");
    assert_eq!(
        prompt_transport.requests.lock().unwrap()[0].url,
        "https://api.minimax.cn/v1/voice_clone"
    );
}

#[tokio::test]
async fn design_sends_required_caller_preview_and_keeps_hex_trial_audio_raw() {
    let trial_audio = "00a1FE7f";
    let native = json!({
        "trial_audio": trial_audio,
        "voice_id":"vdesign-2026",
        "base_resp":base_resp(),
        "future_field":[1,2,3]
    });
    let transport = MockTransport::new([json_response(200, "design-req-1", native.clone())]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let result = voice_service
        .design_voice(
            &MiniMaxVoiceDesignRequest::new("calm low male voice", "This is the paid preview.")
                .with_voice_id("vdesign-2026"),
            &credentials(),
        )
        .await
        .unwrap();
    assert_eq!(result.reference.kind(), MiniMaxVoiceKind::Generated);
    assert_eq!(result.reference.voice_id(), "vdesign-2026");
    assert_eq!(result.trial_audio.as_deref(), Some(trial_audio));
    assert_eq!(result.native, native);
    assert_eq!(result.request_id.as_deref(), Some("design-req-1"));
    let sent = transport.requests.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(sent[0].url, format!("{API_ROOT}/voice_design"));
    assert_eq!(
        body,
        json!({
            "prompt":"calm low male voice",
            "preview_text":"This is the paid preview.",
            "voice_id":"vdesign-2026"
        })
    );
}

#[tokio::test]
async fn mainland_design_watermark_is_typed_and_serialized_only_when_requested() {
    let native = json!({
        "trial_audio":"abc123",
        "voice_id":"cn-designed-voice",
        "base_resp":base_resp()
    });
    let transport = MockTransport::new([json_response(200, "design-cn-1", native)]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::ChinaMainland);
    voice_service
        .design_voice(
            &MiniMaxVoiceDesignRequest::new(
                "悬疑故事\n播音员",
                "这是明确请求的付费试听。\n请保留换行。",
            )
            .with_aigc_watermark(false),
            &credentials(),
        )
        .await
        .unwrap();

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests[0].url, "http://127.0.0.1:8420/v1/voice_design");
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body,
        json!({
            "prompt":"悬疑故事\n播音员",
            "preview_text":"这是明确请求的付费试听。\n请保留换行。",
            "aigc_watermark":false
        })
    );
}

#[tokio::test]
async fn list_parses_three_categories_and_preserves_system_ids_and_unknown_fields() {
    let native = json!({
        "system_voice":[{
            "voice_id":"Chinese (Mandarin)_Reliable_Executive",
            "voice_name":"Steady Executive",
            "description":["A steady and reliable voice."],
            "created_time":"1970-01-01",
            "extension":"system-extra"
        }],
        "voice_cloning":[{
            "voice_id":"Narrator_01",
            "description":[],
            "created_time":"2025-08-20"
        }],
        "voice_generation":[{
            "voice_id":"ttv-voice-2025082011321125-2uEN0X1S",
            "description":[],
            "created_time":"2025-08-20"
        }],
        "base_resp":base_resp(),
        "future_category":{"kept":true}
    });
    let transport = MockTransport::new([json_response(200, "list-req-1", native.clone())]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let list = voice_service
        .list_voices(
            &MiniMaxVoiceListRequest::new(MiniMaxVoiceListType::All),
            &credentials(),
        )
        .await
        .unwrap();
    assert_eq!(list.system_voice.len(), 1);
    assert_eq!(list.voice_cloning.len(), 1);
    assert_eq!(list.voice_generation.len(), 1);
    assert_eq!(
        list.system_voice[0].reference.voice_id(),
        "Chinese (Mandarin)_Reliable_Executive"
    );
    assert_eq!(
        list.system_voice[0].reference.kind(),
        MiniMaxVoiceKind::System
    );
    assert_eq!(
        list.system_voice[0].voice_name.as_deref(),
        Some("Steady Executive")
    );
    assert_eq!(
        list.system_voice[0].description,
        ["A steady and reliable voice."]
    );
    assert_eq!(
        list.voice_cloning[0].reference.kind(),
        MiniMaxVoiceKind::Cloned
    );
    assert_eq!(
        list.voice_generation[0].reference.kind(),
        MiniMaxVoiceKind::Generated
    );
    assert_eq!(list.native, native);
    let sent = transport.requests.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(sent[0].url, format!("{API_ROOT}/get_voice"));
    assert_eq!(body, json!({"voice_type":"all"}));
}

#[tokio::test]
async fn delete_only_accepts_cloned_or_generated_voice_and_binds_response_id() {
    let transport = MockTransport::new([json_response(
        200,
        "delete-req-1",
        json!({"voice_id":"Narrator_01","created_time":"1728962464","base_resp":base_resp()}),
    )]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let reference = MiniMaxVoiceRef::new(
        voice_service.scope().clone(),
        MiniMaxVoiceKind::Cloned,
        "Narrator_01",
    )
    .unwrap();
    let deleted = voice_service
        .delete_voice(&reference, &credentials())
        .await
        .unwrap();
    assert_eq!(deleted.reference, reference);
    assert_eq!(deleted.created_time.as_deref(), Some("1728962464"));
    {
        let sent = transport.requests.lock().unwrap();
        assert_eq!(sent[0].url, format!("{API_ROOT}/delete_voice"));
        assert_eq!(
            serde_json::from_slice::<Value>(&sent[0].body).unwrap(),
            json!({
                "voice_type":"voice_cloning",
                "voice_id":"Narrator_01"
            })
        );
    }

    let denied_transport = MockTransport::new([]);
    let denied_service = service(&denied_transport, MiniMaxVoicesRegion::International);
    let system_voice = MiniMaxVoiceRef::new(
        denied_service.scope().clone(),
        MiniMaxVoiceKind::System,
        "Chinese (Mandarin)_Reliable_Executive",
    )
    .unwrap();
    assert!(matches!(
        denied_service
            .delete_voice(&system_voice, &credentials())
            .await,
        Err(MiniMaxVoicesError::InvalidRequest(_))
    ));
    assert!(denied_transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn file_scope_purpose_expiration_and_documented_input_bounds_are_checked_before_http() {
    let transport = MockTransport::new([]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let mut wrong_purpose = file_ref(&voice_service, "901", "prompt_audio");
    let request = MiniMaxVoiceCloneRequest::new(wrong_purpose.clone(), "Narrator_01");
    assert!(matches!(
        voice_service.clone_voice(&request, &credentials()).await,
        Err(MiniMaxVoicesError::InvalidRequest(_))
            | Err(MiniMaxVoicesError::Llm(LlmError::PermissionDenied { .. }))
    ));

    wrong_purpose.purpose = Some("voice_clone".into());
    wrong_purpose.expires_at = Some("1".into());
    let expired = MiniMaxVoiceCloneRequest::new(wrong_purpose, "Narrator_01");
    assert!(matches!(
        voice_service.clone_voice(&expired, &credentials()).await,
        Err(MiniMaxVoicesError::InvalidRequest(_))
    ));

    let oversized = file_ref(&voice_service, "902", "voice_clone");
    let mut oversized = oversized;
    oversized.size_bytes = Some(20_000_001);
    let oversized_request = MiniMaxVoiceCloneRequest::new(oversized, "Narrator_01");
    assert!(matches!(
        voice_service
            .clone_voice(&oversized_request, &credentials())
            .await,
        Err(MiniMaxVoicesError::InvalidRequest(_))
    ));

    for (id, duration) in [("Narrator_02", 9.0), ("ends_", 60.0)] {
        let request =
            MiniMaxVoiceCloneRequest::new(file_ref(&voice_service, "903", "voice_clone"), id)
                .with_voice_audio_duration_seconds(duration);
        assert!(matches!(
            voice_service.clone_voice(&request, &credentials()).await,
            Err(MiniMaxVoicesError::InvalidRequest(_))
        ));
    }
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn optional_dependencies_accuracy_and_design_preview_are_preflighted() {
    let transport = MockTransport::new([]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let file = file_ref(&voice_service, "904", "voice_clone");
    let requests = [
        MiniMaxVoiceCloneRequest::new(file.clone(), "Narrator_01").with_accuracy(0.7),
        MiniMaxVoiceCloneRequest::new(file.clone(), "Narrator_01")
            .with_preview("preview text", MiniMaxTtsModel::Speech28Hd)
            .with_accuracy(f64::NAN),
        MiniMaxVoiceCloneRequest::new(file, "Narrator_01")
            .with_preview("preview text", MiniMaxTtsModel::Speech28Hd)
            .with_text_validation("expected words")
            .with_accuracy(1.1),
    ];
    for request in requests {
        assert!(matches!(
            voice_service.clone_voice(&request, &credentials()).await,
            Err(MiniMaxVoicesError::InvalidRequest(_))
        ));
    }
    let design = MiniMaxVoiceDesignRequest::new("prompt", "x".repeat(501));
    assert!(matches!(
        voice_service.design_voice(&design, &credentials()).await,
        Err(MiniMaxVoicesError::InvalidRequest(_))
    ));
    let international_watermark =
        MiniMaxVoiceDesignRequest::new("prompt", "preview").with_aigc_watermark(true);
    assert!(matches!(
        voice_service
            .design_voice(&international_watermark, &credentials())
            .await,
        Err(MiniMaxVoicesError::InvalidRequest(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn dispatch_uncertainty_and_malformed_2xx_preserve_request_id_and_native() {
    let transport = MockTransport::new([
        Outcome::Transport(LlmError::TransportTimeout {
            message: "connection timed out after dispatch".into(),
        }),
        json_response(
            200,
            "malformed-ack-1",
            json!({"voice_id":"Narrator_01","provider_extension":{"kept":true}}),
        ),
        json_response(
            500,
            "server-error-1",
            json!({"base_resp":{"status_code":5001,"status_msg":"internal"}}),
        ),
    ]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let request = MiniMaxVoiceCloneRequest::new(
        file_ref(&voice_service, "905", "voice_clone"),
        "Narrator_01",
    );
    assert!(matches!(
        voice_service.clone_voice(&request, &credentials()).await,
        Err(MiniMaxVoicesError::OutcomeUnknown {
            operation: "clone-voice",
            ..
        })
    ));
    match voice_service
        .clone_voice(&request, &credentials())
        .await
        .unwrap_err()
    {
        MiniMaxVoicesError::ResponseOutcomeUnknown {
            operation: "clone-voice",
            request_id,
            native,
            ..
        } => {
            assert_eq!(request_id.as_deref(), Some("malformed-ack-1"));
            assert_eq!(native["provider_extension"]["kept"], true);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    match voice_service
        .clone_voice(&request, &credentials())
        .await
        .unwrap_err()
    {
        MiniMaxVoicesError::Provider {
            dispatch,
            http_status: Some(500),
            ..
        } => assert_eq!(dispatch, MiniMaxVoicesDispatch::Unknown),
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(transport.requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn provider_status_code_errors_are_rejections_not_unknown_successes() {
    let transport = MockTransport::new([json_response(
        200,
        "clone-rejected-1",
        json!({
            "base_resp":{"status_code":1043,"status_msg":"similarity check failed"},
            "provider_info":"retained"
        }),
    )]);
    let voice_service = service(&transport, MiniMaxVoicesRegion::International);
    let request = MiniMaxVoiceCloneRequest::new(
        file_ref(&voice_service, "906", "voice_clone"),
        "Narrator_01",
    );
    let error = voice_service
        .clone_voice(&request, &credentials())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        MiniMaxVoicesError::Provider {
            code: Some(1043),
            dispatch: MiniMaxVoicesDispatch::Rejected,
            request_id: Some(ref id),
            ..
        } if id == "clone-rejected-1"
    ));
}

#[tokio::test]
async fn malformed_voice_category_and_wrong_delete_id_are_reported() {
    let list_transport = MockTransport::new([json_response(
        200,
        "list-bad-1",
        json!({
            "system_voice":[{"voice_id":"voice","description":"unexpected-shape"}],
            "base_resp":base_resp()
        }),
    )]);
    let list_service = service(&list_transport, MiniMaxVoicesRegion::International);
    assert!(matches!(
        list_service
            .list_voices(
                &MiniMaxVoiceListRequest::new(MiniMaxVoiceListType::System),
                &credentials()
            )
            .await,
        Err(MiniMaxVoicesError::InvalidResponse {
            operation: "list-voices",
            ..
        })
    ));

    let delete_transport = MockTransport::new([json_response(
        200,
        "delete-wrong-id-1",
        json!({"voice_id":"different","base_resp":base_resp()}),
    )]);
    let delete_service = service(&delete_transport, MiniMaxVoicesRegion::International);
    let reference = MiniMaxVoiceRef::new(
        delete_service.scope().clone(),
        MiniMaxVoiceKind::Generated,
        "ttv-voice-2026",
    )
    .unwrap();
    assert!(matches!(
        delete_service
            .delete_voice(&reference, &credentials())
            .await,
        Err(MiniMaxVoicesError::ResponseOutcomeUnknown {
            operation: "delete-voice",
            ..
        })
    ));
}

#[tokio::test]
async fn references_are_region_and_endpoint_scoped_and_base_url_region_is_enforced() {
    let transport = MockTransport::new([]);
    let main = service(&transport, MiniMaxVoicesRegion::International);
    let mainland_scope = service(&transport, MiniMaxVoicesRegion::ChinaMainland)
        .scope()
        .clone();
    let other_ref =
        MiniMaxVoiceRef::new(mainland_scope, MiniMaxVoiceKind::Cloned, "Narrator_01").unwrap();
    assert!(matches!(
        main.delete_voice(&other_ref, &credentials()).await,
        Err(MiniMaxVoicesError::InvalidRequest(_))
    ));
    let wrong_region = MiniMaxVoicesService::new(
        &transport,
        MiniMaxVoicesConfig::new(
            "voices-profile",
            "team/account-3",
            MiniMaxVoicesRegion::International,
        )
        .with_api_base_url("https://api.minimax.cn/v1"),
    );
    assert!(matches!(
        wrong_region,
        Err(MiniMaxVoicesError::InvalidRequest(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}
