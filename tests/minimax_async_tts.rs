use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::{LlmError, ProtocolFamily, ProviderId, Secret},
    providers::minimax::async_tts::{
        MiniMaxAsyncTtsAudioFormat, MiniMaxAsyncTtsAudioSetting, MiniMaxAsyncTtsConfig,
        MiniMaxAsyncTtsDispatch, MiniMaxAsyncTtsError, MiniMaxAsyncTtsFileRef,
        MiniMaxAsyncTtsRegion, MiniMaxAsyncTtsRequest, MiniMaxAsyncTtsService,
        MiniMaxAsyncTtsStatus, MiniMaxAsyncTtsTaskRef, MINIMAX_ASYNC_TTS_CHINA_BASE,
        MINIMAX_ASYNC_TTS_INTERNATIONAL_BASE,
    },
    providers::minimax::tts::{MiniMaxTtsModel, MiniMaxTtsPronunciationDict},
    providers::minimax::voices::{
        MiniMaxVoiceKind, MiniMaxVoiceRef, MiniMaxVoicesRegion, MiniMaxVoicesScope,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex, time::Duration};

struct Reply {
    status: u16,
    body: Vec<u8>,
}

enum MockReply {
    Response(Reply),
    TransportError,
}

struct MockTransport {
    replies: Mutex<VecDeque<MockReply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = MockReply>) -> Self {
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
        match self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("queued mock reply")
        {
            MockReply::Response(reply) => Ok(StreamResponse {
                status: reply.status,
                headers: vec![("x-request-id".into(), "header-request-1".into())],
                body: futures::stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
            }),
            MockReply::TransportError => Err(LlmError::TransportTimeout {
                message: "mock timed out after dispatch".into(),
            }),
        }
    }
}

fn response(status: u16, body: Value) -> MockReply {
    MockReply::Response(Reply {
        status,
        body: serde_json::to_vec(&body).unwrap(),
    })
}

fn service(transport: &MockTransport, region: MiniMaxAsyncTtsRegion) -> MiniMaxAsyncTtsService<'_> {
    MiniMaxAsyncTtsService::new(
        transport,
        MiniMaxAsyncTtsConfig::new("minimax-long-text", "team/account-17", region)
            .with_api_base_url("http://127.0.0.1:8418/v1")
            .with_request_timeout(Duration::from_secs(3)),
    )
    .unwrap()
}

fn api_key() -> Secret<String> {
    Secret::new("minimax-test-key".to_owned())
}

fn task_success(task_id: u64, file_id: u64) -> Value {
    json!({
        "task_id": task_id,
        "file_id": file_id,
        "task_token": "task-token-secret",
        "usage_characters": 24,
        "trace_id": "trace-submit-1",
        "base_resp": {"status_code": 0, "status_msg": "success"}
    })
}

fn file_ref(service: &MiniMaxAsyncTtsService<'_>, file_id: &str) -> MiniMaxAsyncTtsFileRef {
    MiniMaxAsyncTtsFileRef {
        scope: service.scope().clone(),
        file_id: file_id.into(),
    }
}

fn task_ref(service: &MiniMaxAsyncTtsService<'_>, task_id: &str) -> MiniMaxAsyncTtsTaskRef {
    MiniMaxAsyncTtsTaskRef {
        scope: service.scope().clone(),
        task_id: task_id.into(),
    }
}

fn text_file_ref(service: &MiniMaxAsyncTtsService<'_>, file_id: &str) -> ProviderFileRef {
    ProviderFileRef {
        provider_id: "minimax".into(),
        profile_name: service.scope().profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint("http://127.0.0.1:8418/v1"),
        account_scope: Some(service.scope().account_scope.clone()),
        protocol: ProtocolFamily::OpenAiChat,
        file_id: file_id.into(),
        uri: None,
        filename: Some("long-text.txt".into()),
        media_type: Some("text/plain".into()),
        size_bytes: None,
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: Some("t2a_async_input".into()),
    }
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

#[tokio::test]
async fn submits_text_with_documented_body_and_secret_redaction() {
    let transport =
        MockTransport::new([response(200, task_success(95157322514444, 95157322514445))]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let key = api_key();
    let mut request = MiniMaxAsyncTtsRequest::new("Hello from async MiniMax", "voice-123");
    request.model = MiniMaxTtsModel::Speech28Turbo;
    request.audio_setting = Some(MiniMaxAsyncTtsAudioSetting {
        channel: 2,
        format: MiniMaxAsyncTtsAudioFormat::Wav,
        ..Default::default()
    });
    request.language_boost = Some("auto".into());
    request.pronunciation_dict = Some(MiniMaxTtsPronunciationDict {
        tone: vec!["MiniMax/Mini Max".into()],
    });
    request.voice_modify = Some(json!({"sound_effects":"spacious_echo"}));

    let submission = service.submit(&request, &key).await.unwrap();
    assert_eq!(submission.task.task_id, "95157322514444");
    assert_eq!(submission.file.as_ref().unwrap().file_id, "95157322514445");
    assert_eq!(submission.usage_characters, Some(24));
    assert_eq!(submission.request_id.as_deref(), Some("trace-submit-1"));
    assert_eq!(
        submission.task_token.as_ref().unwrap().expose_secret(),
        "task-token-secret"
    );
    assert!(submission.native.get("task_token").is_none());
    assert!(!format!("{submission:?}").contains("task-token-secret"));

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "http://127.0.0.1:8418/v1/t2a_async_v2");
    assert_eq!(
        requests[0]
            .headers
            .iter()
            .filter(|(name, _)| name == "content-type")
            .count(),
        1
    );
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer minimax-test-key"));
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "speech-2.8-turbo");
    assert_eq!(body["text"], "Hello from async MiniMax");
    assert!(body.get("text_file_id").is_none());
    assert_eq!(body["voice_setting"]["voice_id"], "voice-123");
    assert_eq!(body["audio_setting"]["audio_sample_rate"], 32_000);
    assert_eq!(body["audio_setting"]["format"], "wav");
    assert_eq!(body["audio_setting"]["channel"], 2);
    assert_eq!(body["language_boost"], "auto");
    assert_eq!(body["pronunciation_dict"]["tone"][0], "MiniMax/Mini Max");
    assert_eq!(body["voice_modify"]["sound_effects"], "spacious_echo");
    assert!(body.get("aigc_watermark").is_none());
}

#[tokio::test]
async fn scoped_voice_is_used_and_mismatched_references_fail_before_http() {
    let transport = MockTransport::new([response(200, task_success(31, 32))]);
    let active_service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let voice = MiniMaxVoiceRef::new(
        voice_reference(
            "minimax",
            "minimax-long-text",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::International,
        )
        .scope,
        MiniMaxVoiceKind::Cloned,
        "created-voice-17",
    )
    .unwrap();
    active_service
        .submit_with_voice(
            &MiniMaxAsyncTtsRequest::new("Use my created voice", "built-in-id"),
            &voice,
            &api_key(),
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&transport.requests.lock().unwrap()[0].body).unwrap();
    assert_eq!(body["voice_setting"]["voice_id"], "created-voice-17");

    let transport = MockTransport::new([]);
    let untouched_service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let mismatched = [
        voice_reference(
            "other-provider",
            "minimax-long-text",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::International,
        ),
        voice_reference(
            "minimax",
            "another-profile",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::International,
        ),
        voice_reference(
            "minimax",
            "minimax-long-text",
            "http://127.0.0.1:8418/v1",
            "another-account",
            MiniMaxVoicesRegion::International,
        ),
        voice_reference(
            "minimax",
            "minimax-long-text",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::ChinaMainland,
        ),
        voice_reference(
            "minimax",
            "minimax-long-text",
            "http://127.0.0.1:8419/v1",
            "team/account-17",
            MiniMaxVoicesRegion::International,
        ),
        voice_reference(
            "minimax",
            "minimax-long-text",
            "http://127.0.0.2:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::International,
        ),
    ];
    for voice in mismatched {
        assert!(matches!(
            untouched_service
                .submit_with_voice(
                    &MiniMaxAsyncTtsRequest::new("Must not be submitted", "built-in-id"),
                    &voice,
                    &api_key(),
                )
                .await,
            Err(MiniMaxAsyncTtsError::InvalidRequest(_))
        ));
    }
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn trailing_slash_api_base_accepts_canonical_voice_scope_without_double_slash_route() {
    let transport = MockTransport::new([response(200, task_success(41, 42))]);
    let service = MiniMaxAsyncTtsService::new(
        &transport,
        MiniMaxAsyncTtsConfig::new(
            "minimax-long-text",
            "team/account-17",
            MiniMaxAsyncTtsRegion::International,
        )
        .with_api_base_url("http://127.0.0.1:8418/v1/"),
    )
    .unwrap();
    let voice = MiniMaxVoiceRef::new(
        voice_reference(
            "minimax",
            "minimax-long-text",
            "http://127.0.0.1:8418/v1",
            "team/account-17",
            MiniMaxVoicesRegion::International,
        )
        .scope,
        MiniMaxVoiceKind::Cloned,
        "created-voice-17",
    )
    .unwrap();

    service
        .submit_with_voice(
            &MiniMaxAsyncTtsRequest::new("Use the same canonical root", "built-in-id"),
            &voice,
            &api_key(),
        )
        .await
        .unwrap();

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests[0].url, "http://127.0.0.1:8418/v1/t2a_async_v2");
    assert!(!requests[0].url.contains("/v1//"));
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["voice_setting"]["voice_id"], "created-voice-17");
}

#[tokio::test]
async fn text_file_input_is_scoped_and_sent_as_an_int64() {
    let transport = MockTransport::new([response(200, task_success(22, 23))]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::ChinaMainland);
    let request = MiniMaxAsyncTtsRequest::from_text_file(
        text_file_ref(&service, "900719925474099"),
        "voice-cn",
    );
    service.submit(&request, &api_key()).await.unwrap();

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests[0].url, "http://127.0.0.1:8418/v1/t2a_async_v2");
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["text_file_id"], 900719925474099_u64);
    assert!(body.get("text").is_none());
}

#[tokio::test]
async fn queries_once_and_preserves_unknown_status_without_auto_polling() {
    let transport = MockTransport::new([response(
        200,
        json!({
            "task_id": 123456,
            "status": "FutureStatus",
            "file_id": 789,
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }),
    )]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let task = service
        .query(&task_ref(&service, "123456"), &api_key())
        .await
        .unwrap();

    assert_eq!(
        task.status,
        MiniMaxAsyncTtsStatus::Other("FutureStatus".into())
    );
    assert_eq!(task.file.unwrap().file_id, "789");
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(
        requests[0].url,
        "http://127.0.0.1:8418/v1/query/t2a_async_query_v2?task_id=123456"
    );
}

#[tokio::test]
async fn retrieves_result_metadata_and_redacts_the_temporary_url() {
    let signed_url = "https://cdn.minimax.example/audio?signature=do-not-log";
    let transport = MockTransport::new([response(
        200,
        json!({
            "file": {
                "file_id": 89,
                "bytes": 1024,
                "created_at": 1700469398,
                "filename": "speech.mp3",
                "purpose": "t2a_async",
                "download_url": signed_url
            },
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }),
    )]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::ChinaMainland);
    let result = service
        .get_result(&file_ref(&service, "89"), &api_key())
        .await
        .unwrap();

    assert_eq!(result.download_url.as_str(), signed_url);
    assert_eq!(
        result.download_url_valid_for,
        Duration::from_secs(9 * 60 * 60)
    );
    assert_eq!(result.filename.as_deref(), Some("speech.mp3"));
    assert_eq!(result.size_bytes, Some(1024));
    assert_eq!(result.created_at, Some(1700469398));
    assert_eq!(result.purpose.as_deref(), Some("t2a_async"));
    assert_eq!(result.native["file"]["file_id"], 89);
    assert!(result.native["file"].get("download_url").is_none());
    assert!(!format!("{result:?}").contains("do-not-log"));

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(
        requests[0].url,
        "http://127.0.0.1:8418/v1/files/retrieve?file_id=89"
    );
}

#[tokio::test]
async fn wrong_profile_or_account_references_are_rejected_before_http() {
    let transport = MockTransport::new([]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let mut foreign_task = task_ref(&service, "123");
    foreign_task.scope.account_scope = "other-account".into();
    assert!(matches!(
        service.query(&foreign_task, &api_key()).await,
        Err(MiniMaxAsyncTtsError::Llm(LlmError::PermissionDenied { .. }))
    ));

    let mut foreign_file = file_ref(&service, "456");
    foreign_file.scope.profile_name = "other-profile".into();
    assert!(matches!(
        service.get_result(&foreign_file, &api_key()).await,
        Err(MiniMaxAsyncTtsError::Llm(LlmError::PermissionDenied { .. }))
    ));

    let mut request =
        MiniMaxAsyncTtsRequest::from_text_file(text_file_ref(&service, "900"), "voice");
    if let lingxi_llm_client::providers::minimax::async_tts::MiniMaxAsyncTtsInput::TextFile(file) =
        &mut request.input
    {
        file.account_scope = Some("other-account".into());
    }
    assert!(matches!(
        service.submit(&request, &api_key()).await,
        Err(MiniMaxAsyncTtsError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn local_validation_rejects_oversize_text_before_submit() {
    let transport = MockTransport::new([]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let request = MiniMaxAsyncTtsRequest::new("x".repeat(50_001), "voice");
    assert!(matches!(
        service.submit(&request, &api_key()).await,
        Err(MiniMaxAsyncTtsError::InvalidRequest(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn ambiguous_submit_is_never_retried() {
    let transport = MockTransport::new([MockReply::TransportError]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let error = service
        .submit(&MiniMaxAsyncTtsRequest::new("Hello", "voice"), &api_key())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), MiniMaxAsyncTtsDispatch::Unknown);
    assert!(matches!(error, MiniMaxAsyncTtsError::OutcomeUnknown { .. }));
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn successful_but_malformed_submit_response_is_unknown() {
    let transport = MockTransport::new([response(
        200,
        json!({
            "task_token":"must-not-leak",
            "base_resp":{"status_code":0,"status_msg":"success"}
        }),
    )]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let error = service
        .submit(&MiniMaxAsyncTtsRequest::new("Hello", "voice"), &api_key())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), MiniMaxAsyncTtsDispatch::Unknown);
    assert!(!format!("{error:?}").contains("must-not-leak"));
    assert!(matches!(
        error,
        MiniMaxAsyncTtsError::ResponseOutcomeUnknown { .. }
    ));
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn malformed_result_response_redacts_a_signed_url() {
    let transport = MockTransport::new([response(
        200,
        json!({
            "file": {
                "file_id": 900,
                "download_url": "https://cdn.minimax.example/?signature=must-not-leak"
            },
            "base_resp":{"status_code":0,"status_msg":"success"}
        }),
    )]);
    let service = service(&transport, MiniMaxAsyncTtsRegion::International);
    let error = service
        .get_result(&file_ref(&service, "901"), &api_key())
        .await
        .unwrap_err();
    assert!(matches!(
        &error,
        MiniMaxAsyncTtsError::InvalidResponse { .. }
    ));
    assert!(!format!("{error:?}").contains("must-not-leak"));
}

#[tokio::test]
async fn submit_server_error_is_unknown_but_client_rejection_is_definite() {
    let server = MockTransport::new([response(
        503,
        json!({"base_resp":{"status_code":5001,"status_msg":"busy"}}),
    )]);
    let server_service = service(&server, MiniMaxAsyncTtsRegion::International);
    let error = server_service
        .submit(&MiniMaxAsyncTtsRequest::new("Hello", "voice"), &api_key())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), MiniMaxAsyncTtsDispatch::Unknown);
    assert_eq!(server.requests.lock().unwrap().len(), 1);

    let client = MockTransport::new([response(
        400,
        json!({"base_resp":{"status_code":1002,"status_msg":"invalid request"}}),
    )]);
    let client_service = service(&client, MiniMaxAsyncTtsRegion::International);
    let error = client_service
        .submit(&MiniMaxAsyncTtsRequest::new("Hello", "voice"), &api_key())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), MiniMaxAsyncTtsDispatch::Rejected);
    assert!(matches!(error, MiniMaxAsyncTtsError::Provider { .. }));
}

#[test]
fn default_routes_and_region_binding_match_official_hosts() {
    assert_eq!(
        MiniMaxAsyncTtsConfig::new("p", "a", MiniMaxAsyncTtsRegion::International).api_base_url,
        MINIMAX_ASYNC_TTS_INTERNATIONAL_BASE
    );
    assert_eq!(
        MiniMaxAsyncTtsConfig::new("p", "a", MiniMaxAsyncTtsRegion::ChinaMainland).api_base_url,
        MINIMAX_ASYNC_TTS_CHINA_BASE
    );

    let transport = MockTransport::new([]);
    let wrong_region = MiniMaxAsyncTtsService::new(
        &transport,
        MiniMaxAsyncTtsConfig::new("p", "a", MiniMaxAsyncTtsRegion::ChinaMainland)
            .with_api_base_url(MINIMAX_ASYNC_TTS_INTERNATIONAL_BASE),
    );
    assert!(matches!(
        wrong_region,
        Err(MiniMaxAsyncTtsError::InvalidRequest(_))
    ));
}
