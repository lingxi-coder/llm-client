use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::mpsc,
    executor::block_on,
    stream::{self, BoxStream},
    FutureExt, StreamExt,
};
use lingxi_llm_client::{
    files::provider_file_endpoint_fingerprint,
    protocol::{ProviderId, Secret},
    providers::minimax::streaming_tts::{
        MiniMaxStreamingTtsAudioFormat, MiniMaxStreamingTtsConfig, MiniMaxStreamingTtsEmotion,
        MiniMaxStreamingTtsError, MiniMaxStreamingTtsEvent, MiniMaxStreamingTtsLimits,
        MiniMaxStreamingTtsModel, MiniMaxStreamingTtsRequest, MiniMaxStreamingTtsService,
        MiniMaxStreamingTtsSoundEffect, MiniMaxStreamingTtsSubtitleType,
        MiniMaxStreamingTtsTimbreWeight, MiniMaxStreamingTtsVoiceModify,
        MiniMaxStreamingTtsVoiceSetting, MINIMAX_STREAMING_TTS_CHINA_ENDPOINT,
        MINIMAX_STREAMING_TTS_INTERNATIONAL_ENDPOINT,
    },
    providers::minimax::tts::MiniMaxTtsRegion,
    providers::minimax::voices::{
        MiniMaxVoiceKind, MiniMaxVoiceLanguageBoost, MiniMaxVoiceRef, MiniMaxVoicesRegion,
        MiniMaxVoicesScope,
    },
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeSink, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

type Incoming = Result<RealtimeFrame, RealtimeError>;

#[derive(Clone, Copy)]
enum ContinueBehavior {
    Audio,
    FailWrite,
    TaskFailed,
    PartialThenClose,
    PendingContinue,
    PendingFinish,
}

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
    continue_behavior: ContinueBehavior,
    close_count: Arc<Mutex<usize>>,
}

struct FakePeer {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
    close_count: Arc<Mutex<usize>>,
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    continue_behavior: ContinueBehavior,
    incoming: Option<mpsc::UnboundedSender<Incoming>>,
    close_count: Arc<Mutex<usize>>,
}

fn fake_transport(continue_behavior: ContinueBehavior) -> (Arc<FakeTransport>, FakePeer) {
    let connected = stream::iter([Ok(RealtimeFrame::text(
        json!({
            "event": "connected_success",
            "session_id": "session-1",
            "trace_id": "trace-connect",
            "base_resp": { "status_code": 0, "status_msg": "success" }
        })
        .to_string(),
    ))]);
    let sent = Arc::new(Mutex::new(Vec::new()));
    let request = Arc::new(Mutex::new(None));
    let connect_count = Arc::new(Mutex::new(0));
    let close_count = Arc::new(Mutex::new(0));
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(connected.boxed())),
            sent: sent.clone(),
            request: request.clone(),
            connect_count: connect_count.clone(),
            continue_behavior,
            close_count: close_count.clone(),
        }),
        FakePeer {
            sent,
            request,
            connect_count,
            close_count,
        },
    )
}

#[async_trait]
impl RealtimeTransport for FakeTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        *self.request.lock().unwrap() = Some(request);
        *self.connect_count.lock().unwrap() += 1;
        let incoming = self
            .incoming
            .lock()
            .unwrap()
            .take()
            .expect("one connection per fake transport");
        let (responses, response_stream) = mpsc::unbounded();
        let incoming = incoming.chain(response_stream).boxed();
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
                incoming: Some(responses),
                continue_behavior: self.continue_behavior,
                close_count: self.close_count.clone(),
            }),
            inbound: incoming,
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        let RealtimeFrame::Text(bytes) = &frame else {
            return Err(RealtimeError::Transport {
                message: "expected JSON text frame".into(),
            });
        };
        let event: Value = serde_json::from_slice(bytes).unwrap();
        self.sent.lock().unwrap().push(frame);
        if (event["event"] == "task_continue"
            && matches!(self.continue_behavior, ContinueBehavior::PendingContinue))
            || (event["event"] == "task_finish"
                && matches!(self.continue_behavior, ContinueBehavior::PendingFinish))
        {
            // The frame has reached the transport, but its write/flush has
            // not resolved. Dropping this future models timeout/select cancel.
            futures::future::pending::<()>().await;
        }
        match event["event"].as_str() {
            Some("task_start") => {
                self.respond(json!({
                    "event": "task_started",
                    "session_id": "session-1",
                    "trace_id": "trace-start",
                    "base_resp": { "status_code": 0, "status_msg": "success" }
                }));
            }
            Some("task_continue")
                if matches!(self.continue_behavior, ContinueBehavior::FailWrite) =>
            {
                return Err(RealtimeError::Transport {
                    message: "injected uncertain socket write".into(),
                });
            }
            Some("task_continue")
                if matches!(self.continue_behavior, ContinueBehavior::TaskFailed) =>
            {
                self.respond(json!({
                    "event": "task_failed",
                    "session_id": "session-1",
                    "trace_id": "trace-failed",
                    "base_resp": { "status_code": 1004, "status_msg": "task failed" }
                }));
            }
            Some("task_continue")
                if matches!(self.continue_behavior, ContinueBehavior::PartialThenClose) =>
            {
                self.respond(json!({
                    "data": { "audio": "4869" },
                    "is_final": false,
                    "session_id": "session-1",
                    "trace_id": "trace-partial",
                    "base_resp": { "status_code": 0, "status_msg": "success" }
                }));
                self.incoming.take();
            }
            Some("task_continue") => {
                self.respond(json!({
                    "data": { "audio": "4869" },
                    "is_final": false,
                    "session_id": "session-1",
                    "trace_id": "trace-audio-1",
                    "extra_info": { "audio_format": "mp3" },
                    "base_resp": { "status_code": 0, "status_msg": "success" }
                }));
                self.respond(json!({
                    "data": { "audio": "21" },
                    "is_final": true,
                    "session_id": "session-1",
                    "trace_id": "trace-audio-2",
                    "base_resp": { "status_code": 0, "status_msg": "success" }
                }));
            }
            Some("task_finish") => {
                self.respond(json!({
                    "event": "task_finished",
                    "session_id": "session-1",
                    "trace_id": "trace-finish",
                    "base_resp": { "status_code": 0, "status_msg": "success" }
                }));
            }
            _ => {}
        }
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        *self.close_count.lock().unwrap() += 1;
        Ok(())
    }
}

impl FakeSink {
    fn respond(&mut self, value: Value) {
        if let Some(incoming) = &mut self.incoming {
            let _ = incoming.unbounded_send(Ok(RealtimeFrame::text(value.to_string())));
        }
    }
}

fn service(
    transport: Arc<dyn RealtimeTransport>,
) -> Result<MiniMaxStreamingTtsService, MiniMaxStreamingTtsError> {
    service_for_region(transport, MiniMaxTtsRegion::International)
}

fn service_for_region(
    transport: Arc<dyn RealtimeTransport>,
    region: MiniMaxTtsRegion,
) -> Result<MiniMaxStreamingTtsService, MiniMaxStreamingTtsError> {
    MiniMaxStreamingTtsService::new(
        transport,
        MiniMaxStreamingTtsConfig::new("minimax-prod", "account-17", region),
    )
}

fn transport(value: Arc<FakeTransport>) -> Arc<dyn RealtimeTransport> {
    value
}

fn request() -> MiniMaxStreamingTtsRequest {
    let mut request = MiniMaxStreamingTtsRequest::new("English_expressive_narrator");
    request.model = MiniMaxStreamingTtsModel::Speech28Turbo;
    request.voice_setting = MiniMaxStreamingTtsVoiceSetting {
        voice_id: "English_expressive_narrator".into(),
        speed: Some(1.0),
        vol: Some(1.0),
        pitch: Some(0),
        emotion: None,
        english_normalization: None,
        latex_read: None,
    };
    request
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

fn limits() -> MiniMaxStreamingTtsLimits {
    MiniMaxStreamingTtsLimits {
        max_frame_bytes: 16 * 1024,
        max_text_bytes_per_segment: 1024,
        max_pending_segments: 2,
        max_audio_bytes_per_session: 64,
    }
}

fn sent_json(peer: &FakePeer) -> Vec<Value> {
    peer.sent
        .lock()
        .unwrap()
        .iter()
        .map(|frame| match frame {
            RealtimeFrame::Text(bytes) => serde_json::from_slice(bytes).unwrap(),
            RealtimeFrame::Binary(_) => panic!("MiniMax T2A uses JSON text events"),
        })
        .collect()
}

#[test]
fn setup_sends_documented_task_start_and_streams_bounded_hex_audio() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service(transport(fake)).unwrap();
        let session = service
            .connect(
                &Secret::from("per-call-key".to_owned()),
                &request(),
                limits(),
            )
            .await
            .unwrap();
        assert_eq!(
            session.metadata().scope.region,
            MiniMaxTtsRegion::International
        );
        assert_eq!(
            session.metadata().connected_native["event"],
            "connected_success"
        );
        assert_eq!(
            session.metadata().task_started_native["event"],
            "task_started"
        );

        let (mut input, mut events) = session.into_parts();
        input.send_text("first segment").await.unwrap();
        input.send_text("second segment").await.unwrap();
        assert!(matches!(
            input.send_text("third segment").await,
            Err(MiniMaxStreamingTtsError::PendingQueueFull { max: 2 })
        ));
        let mut deltas = Vec::new();
        for _ in 0..4 {
            let Some(MiniMaxStreamingTtsEvent::AudioDelta {
                audio,
                is_final,
                native,
                ..
            }) = events.next().await.unwrap()
            else {
                panic!("expected audio delta")
            };
            deltas.push((audio, is_final, native));
        }
        assert_eq!(deltas[0].0, Bytes::from_static(b"Hi"));
        assert!(!deltas[0].1);
        assert_eq!(deltas[1].0, Bytes::from_static(b"!"));
        assert!(deltas[1].1);
        assert_eq!(deltas[0].2["trace_id"], "trace-audio-1");
        assert_eq!(deltas[2].0, Bytes::from_static(b"Hi"));
        assert!(deltas[3].1);

        input.finish().await.unwrap();
        assert!(matches!(
            events.next().await.unwrap(),
            Some(MiniMaxStreamingTtsEvent::TaskFinished { .. })
        ));
        assert_eq!(events.next().await.unwrap(), None);

        let request = peer.request.lock().unwrap();
        let request = request.as_ref().unwrap();
        assert_eq!(
            request.endpoint,
            MINIMAX_STREAMING_TTS_INTERNATIONAL_ENDPOINT
        );
        assert_eq!(request.max_frame_bytes, limits().max_frame_bytes);
        assert_eq!(
            request.headers,
            [("Authorization".into(), "Bearer per-call-key".into())]
        );
        assert!(!format!("{request:?}").contains("per-call-key"));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);

        let sent = sent_json(&peer);
        assert_eq!(sent[0]["event"], "task_start");
        assert_eq!(sent[0]["model"], "speech-2.8-turbo");
        assert_eq!(
            sent[0]["voice_setting"]["voice_id"],
            "English_expressive_narrator"
        );
        assert_eq!(sent[0]["audio_setting"]["format"], "mp3");
        assert_eq!(
            sent[1],
            json!({ "event": "task_continue", "text": "first segment" })
        );
        assert_eq!(
            sent[2],
            json!({ "event": "task_continue", "text": "second segment" })
        );
        assert_eq!(sent[3], json!({ "event": "task_finish" }));
    });
}

#[test]
fn scoped_voice_is_used_and_mismatched_references_fail_before_wss() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let active_service = service(transport(fake)).unwrap();
        let voice = MiniMaxVoiceRef::new(
            voice_reference(
                "minimax",
                "minimax-prod",
                "https://api.minimax.io/v1",
                "account-17",
                MiniMaxVoicesRegion::International,
            )
            .scope,
            MiniMaxVoiceKind::Cloned,
            "created-voice-17",
        )
        .unwrap();
        let _session = active_service
            .connect_with_voice(
                &Secret::from("per-call-key".to_owned()),
                &request(),
                &voice,
                limits(),
            )
            .await
            .unwrap();
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
        assert_eq!(
            sent_json(&peer)[0]["voice_setting"]["voice_id"],
            "created-voice-17"
        );

        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let untouched_service = service(transport(fake)).unwrap();
        let mismatched = [
            voice_reference(
                "other-provider",
                "minimax-prod",
                "https://api.minimax.io/v1",
                "account-17",
                MiniMaxVoicesRegion::International,
            ),
            voice_reference(
                "minimax",
                "another-profile",
                "https://api.minimax.io/v1",
                "account-17",
                MiniMaxVoicesRegion::International,
            ),
            voice_reference(
                "minimax",
                "minimax-prod",
                "https://api.minimax.io/v1",
                "another-account",
                MiniMaxVoicesRegion::International,
            ),
            voice_reference(
                "minimax",
                "minimax-prod",
                "https://api.minimax.io/v1",
                "account-17",
                MiniMaxVoicesRegion::ChinaMainland,
            ),
            voice_reference(
                "minimax",
                "minimax-prod",
                "https://api.minimax.io:444/v1",
                "account-17",
                MiniMaxVoicesRegion::International,
            ),
        ];
        for voice in mismatched {
            assert!(matches!(
                untouched_service
                    .connect_with_voice(
                        &Secret::from("per-call-key".to_owned()),
                        &request(),
                        &voice,
                        limits(),
                    )
                    .await,
                Err(MiniMaxStreamingTtsError::InvalidInput { .. })
            ));
        }
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
    });
}

#[test]
fn mainland_route_is_selected_and_uses_the_documented_native_handshake() {
    block_on(async {
        let config = MiniMaxStreamingTtsConfig::new(
            "minimax-prod",
            "account-17",
            MiniMaxTtsRegion::ChinaMainland,
        );
        assert_eq!(config.endpoint, MINIMAX_STREAMING_TTS_CHINA_ENDPOINT);

        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service_for_region(transport(fake), MiniMaxTtsRegion::ChinaMainland).unwrap();
        assert_eq!(service.scope().region, MiniMaxTtsRegion::ChinaMainland);
        service
            .connect(&Secret::from("cn-key".to_owned()), &request(), limits())
            .await
            .unwrap();

        let connect = peer.request.lock().unwrap().clone().unwrap();
        assert_eq!(connect.endpoint, "wss://api.minimax.cn/ws/v1/t2a_v2");
        assert_eq!(connect.headers[0].1, "Bearer cn-key");
        let sent = sent_json(&peer);
        assert_eq!(sent[0]["event"], "task_start");
        assert_eq!(sent[0]["model"], "speech-2.8-turbo");
        assert_eq!(
            sent[0]["voice_setting"]["voice_id"],
            "English_expressive_narrator"
        );
        assert_eq!(sent[0]["audio_setting"]["format"], "mp3");
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
    });
}

#[test]
fn route_and_voice_reference_region_mismatches_fail_before_transport() {
    block_on(async {
        let (fake, _peer) = fake_transport(ContinueBehavior::Audio);
        let international_with_china_route = MiniMaxStreamingTtsService::new(
            transport(fake.clone()),
            MiniMaxStreamingTtsConfig::new(
                "minimax-prod",
                "account-17",
                MiniMaxTtsRegion::International,
            )
            .with_endpoint(MINIMAX_STREAMING_TTS_CHINA_ENDPOINT),
        );
        assert!(international_with_china_route.is_err());
        let mainland_with_international_route = MiniMaxStreamingTtsService::new(
            transport(fake.clone()),
            MiniMaxStreamingTtsConfig::new(
                "minimax-prod",
                "account-17",
                MiniMaxTtsRegion::ChinaMainland,
            )
            .with_endpoint(MINIMAX_STREAMING_TTS_INTERNATIONAL_ENDPOINT),
        );
        assert!(mainland_with_international_route.is_err());
        let unconfirmed = MiniMaxStreamingTtsService::new(
            transport(fake),
            MiniMaxStreamingTtsConfig::new(
                "minimax-prod",
                "account-17",
                MiniMaxTtsRegion::ChinaMainland,
            )
            .with_endpoint("wss://api.minimax.cn/ws/v1/t2a_v2?debug=1"),
        );
        assert!(unconfirmed.is_err());

        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let mainland =
            service_for_region(transport(fake), MiniMaxTtsRegion::ChinaMainland).unwrap();
        let wrong_region_ref = voice_reference(
            "minimax",
            "minimax-prod",
            "https://api.minimax.io/v1",
            "account-17",
            MiniMaxVoicesRegion::International,
        );
        assert!(matches!(
            mainland
                .connect_with_voice(
                    &Secret::from("cn-key".to_owned()),
                    &request(),
                    &wrong_region_ref,
                    limits(),
                )
                .await,
            Err(MiniMaxStreamingTtsError::InvalidInput { .. })
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
    });
}

#[test]
fn mainland_scoped_voice_uses_the_matching_regional_api_root() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let mainland =
            service_for_region(transport(fake), MiniMaxTtsRegion::ChinaMainland).unwrap();
        let voice = MiniMaxVoiceRef::new(
            voice_reference(
                "minimax",
                "minimax-prod",
                "https://api.minimax.cn/v1",
                "account-17",
                MiniMaxVoicesRegion::ChinaMainland,
            )
            .scope,
            MiniMaxVoiceKind::Cloned,
            "created-voice-cn",
        )
        .unwrap();
        mainland
            .connect_with_voice(
                &Secret::from("cn-key".to_owned()),
                &request(),
                &voice,
                limits(),
            )
            .await
            .unwrap();
        assert_eq!(
            sent_json(&peer)[0]["voice_setting"]["voice_id"],
            "created-voice-cn"
        );
        assert_eq!(
            peer.request.lock().unwrap().as_ref().unwrap().endpoint,
            MINIMAX_STREAMING_TTS_CHINA_ENDPOINT
        );
    });
}

#[test]
fn complete_task_start_schema_is_typed_and_preserved_on_the_wire() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service(transport(fake)).unwrap();
        let mut request = request();
        request.model = MiniMaxStreamingTtsModel::Speech28Hd;
        request.voice_setting.voice_id.clear();
        request.voice_setting.emotion = Some(MiniMaxStreamingTtsEmotion::Happy);
        request.voice_setting.english_normalization = Some(false);
        request.voice_setting.latex_read = Some(true);
        request.audio_setting.channel = 2;
        request.language_boost = Some(MiniMaxVoiceLanguageBoost::Chinese);
        request.pronunciation_dict = Some(
            lingxi_llm_client::providers::minimax::tts::MiniMaxTtsPronunciationDict {
                tone: vec!["word/(wo3)(rd1)".into()],
            },
        );
        request.timbre_weights = Some(vec![
            MiniMaxStreamingTtsTimbreWeight {
                voice_id: "voice-a".into(),
                weight: 30,
            },
            MiniMaxStreamingTtsTimbreWeight {
                voice_id: "voice-b".into(),
                weight: 70,
            },
        ]);
        request.voice_modify = Some(MiniMaxStreamingTtsVoiceModify {
            pitch: Some(-12),
            intensity: Some(30),
            timbre: Some(8),
            sound_effects: Some(MiniMaxStreamingTtsSoundEffect::SpaciousEcho),
        });
        request.subtitle_enable = Some(true);
        request.subtitle_type = Some(MiniMaxStreamingTtsSubtitleType::WordStreaming);
        request.continuous_sound = Some(true);

        service
            .connect(&Secret::from("per-call-key".to_owned()), &request, limits())
            .await
            .unwrap();
        let sent = sent_json(&peer);
        let start = &sent[0];
        assert_eq!(start.as_object().unwrap().len(), 11);
        assert_eq!(start["voice_setting"]["voice_id"], "");
        assert_eq!(start["voice_setting"]["emotion"], "happy");
        assert_eq!(start["voice_setting"]["english_normalization"], false);
        assert_eq!(start["voice_setting"]["latex_read"], true);
        assert_eq!(start["audio_setting"]["channel"], 2);
        assert_eq!(start["language_boost"], "Chinese");
        assert_eq!(start["pronunciation_dict"]["tone"][0], "word/(wo3)(rd1)");
        assert_eq!(start["timbre_weights"][1]["weight"], 70);
        assert_eq!(start["voice_modify"]["sound_effects"], "spacious_echo");
        assert_eq!(start["subtitle_type"], "word_streaming");
        assert_eq!(start["continuous_sound"], true);
    });
}

#[test]
fn task_start_model_and_setting_combinations_fail_before_transport() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service(transport(fake)).unwrap();
        let mut invalid_requests = Vec::new();

        let mut bad_request = request();
        bad_request.voice_setting.emotion = Some(MiniMaxStreamingTtsEmotion::Whisper);
        invalid_requests.push(bad_request);

        let mut bad_request = request();
        bad_request.model = MiniMaxStreamingTtsModel::Speech26Hd;
        bad_request.continuous_sound = Some(false);
        invalid_requests.push(bad_request);

        let mut bad_request = request();
        bad_request.voice_modify = Some(MiniMaxStreamingTtsVoiceModify {
            pitch: Some(0),
            intensity: None,
            timbre: None,
            sound_effects: None,
        });
        bad_request.audio_setting.format = MiniMaxStreamingTtsAudioFormat::Flac;
        invalid_requests.push(bad_request);

        let mut bad_request = request();
        bad_request.voice_setting.latex_read = Some(true);
        bad_request.language_boost = Some(MiniMaxVoiceLanguageBoost::French);
        invalid_requests.push(bad_request);

        let mut bad_request = request();
        bad_request.model = MiniMaxStreamingTtsModel::Speech02Hd;
        bad_request.language_boost = Some(MiniMaxVoiceLanguageBoost::Persian);
        invalid_requests.push(bad_request);

        let mut bad_request = request();
        bad_request.timbre_weights = Some(vec![MiniMaxStreamingTtsTimbreWeight {
            voice_id: "voice-a".into(),
            weight: 100,
        }]);
        invalid_requests.push(bad_request);

        for invalid_request in invalid_requests {
            assert!(matches!(
                service
                    .connect(
                        &Secret::from("per-call-key".to_owned()),
                        &invalid_request,
                        limits(),
                    )
                    .await,
                Err(MiniMaxStreamingTtsError::InvalidInput { .. })
            ));
        }
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
    });
}

#[test]
fn non_mp3_task_start_omits_mp3_only_bitrate_setting() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service(transport(fake)).unwrap();
        let mut request = request();
        request.audio_setting.format = MiniMaxStreamingTtsAudioFormat::Opus;
        request.audio_setting.sample_rate = 44_100;
        service
            .connect(&Secret::from("per-call-key".to_owned()), &request, limits())
            .await
            .unwrap();
        let sent = sent_json(&peer);
        assert_eq!(sent[0]["audio_setting"]["format"], "opus");
        assert!(sent[0]["audio_setting"].get("bitrate").is_none());
    });
}

#[test]
fn uncertain_text_write_is_not_retried() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::FailWrite);
        let service = service(transport(fake)).unwrap();
        let (mut input, _events) = service
            .connect(
                &Secret::from("per-call-key".to_owned()),
                &request(),
                limits(),
            )
            .await
            .unwrap()
            .into_parts();
        assert!(matches!(
            input.send_text("one attempt").await,
            Err(MiniMaxStreamingTtsError::OutcomeUnknown {
                operation: "task_continue",
                ..
            })
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
        assert_eq!(
            sent_json(&peer)
                .iter()
                .filter(|event| event["event"] == "task_continue")
                .count(),
            1
        );
    });
}

#[test]
fn cancelled_writes_invalidate_the_sender_without_another_transport_write() {
    block_on(async {
        for behavior in [
            ContinueBehavior::PendingContinue,
            ContinueBehavior::PendingFinish,
        ] {
            let (fake, peer) = fake_transport(behavior);
            let service = service(transport(fake)).unwrap();
            let (mut input, mut events) = service
                .connect(&Secret::from("key".to_owned()), &request(), limits())
                .await
                .unwrap()
                .into_parts();
            if matches!(behavior, ContinueBehavior::PendingContinue) {
                assert!(input
                    .send_text("possibly received")
                    .now_or_never()
                    .is_none());
            } else {
                assert!(input.finish().now_or_never().is_none());
            }
            let sent_count = sent_json(&peer).len();
            assert_eq!(sent_count, 2); // task_start and the cancelled write
            assert!(matches!(
                input.send_text("must not replay").await,
                Err(MiniMaxStreamingTtsError::Closed)
            ));
            assert!(matches!(
                input.finish().await,
                Err(MiniMaxStreamingTtsError::Closed)
            ));
            assert_eq!(sent_json(&peer).len(), sent_count);
            // Dropping the owned sink releases its inbound sender. Reading
            // observes interruption rather than hanging on a retained sink.
            assert!(matches!(
                events.next().now_or_never(),
                Some(Err(MiniMaxStreamingTtsError::Interrupted { .. }))
            ));
        }
    });
}

#[test]
fn dropping_an_unpolled_write_does_not_invalidate_the_sender() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service(transport(fake)).unwrap();
        let (mut input, _events) = service
            .connect(&Secret::from("key".to_owned()), &request(), limits())
            .await
            .unwrap()
            .into_parts();
        drop(input.send_text("never sent"));
        drop(input.finish());
        assert!(input.send_text("").await.is_err());
        input.send_text("valid segment").await.unwrap();
        assert_eq!(sent_json(&peer).len(), 2);
    });
}

#[test]
fn audio_limit_closes_the_stream_without_buffering_more_output() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service(transport(fake)).unwrap();
        let mut limits = limits();
        limits.max_audio_bytes_per_session = 1;
        let (mut input, mut events) = service
            .connect(&Secret::from("per-call-key".to_owned()), &request(), limits)
            .await
            .unwrap()
            .into_parts();
        input.send_text("bounded output").await.unwrap();
        assert!(matches!(
            events.next().await,
            Err(MiniMaxStreamingTtsError::AudioLimitExceeded { max: 1 })
        ));
        assert_eq!(*peer.close_count.lock().unwrap(), 1);
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
    });
}

#[test]
fn provider_task_failure_is_preserved_and_closes_the_socket() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::TaskFailed);
        let service = service(transport(fake)).unwrap();
        let (mut input, mut events) = service
            .connect(
                &Secret::from("per-call-key".to_owned()),
                &request(),
                limits(),
            )
            .await
            .unwrap()
            .into_parts();
        input.send_text("fail this task").await.unwrap();
        let Some(MiniMaxStreamingTtsEvent::TaskFailed {
            status_code,
            native,
            ..
        }) = events.next().await.unwrap()
        else {
            panic!("expected the native task_failed event")
        };
        assert_eq!(status_code, Some(1004));
        assert_eq!(native["trace_id"], "trace-failed");
        assert_eq!(*peer.close_count.lock().unwrap(), 1);
    });
}

#[test]
fn disconnect_keeps_yielded_audio_and_returns_last_provider_refs() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::PartialThenClose);
        let service = service(transport(fake)).unwrap();
        let (mut input, mut events) = service
            .connect(
                &Secret::from("per-call-key".to_owned()),
                &request(),
                limits(),
            )
            .await
            .unwrap()
            .into_parts();
        input.send_text("stream then disconnect").await.unwrap();
        assert!(matches!(
            events.next().await.unwrap(),
            Some(MiniMaxStreamingTtsEvent::AudioDelta {
                ref audio,
                is_final: false,
                ..
            }) if audio == &Bytes::from_static(b"Hi")
        ));
        assert!(matches!(
            events.next().await,
            Err(MiniMaxStreamingTtsError::Interrupted {
                session_id: Some(ref session_id),
                trace_id: Some(ref trace_id),
                ..
            }) if session_id == "session-1" && trace_id == "trace-partial"
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
        assert_eq!(
            sent_json(&peer)
                .iter()
                .filter(|event| event["event"] == "task_continue")
                .count(),
            1
        );
    });
}

#[test]
fn limits_and_credentials_fail_before_opening_the_socket() {
    block_on(async {
        let (fake, peer) = fake_transport(ContinueBehavior::Audio);
        let service = service(transport(fake)).unwrap();
        let mut invalid_request = request();
        invalid_request.voice_setting.voice_id.clear();
        assert!(service
            .connect(
                &Secret::from("per-call-key".to_owned()),
                &invalid_request,
                limits()
            )
            .await
            .is_err());
        assert!(service
            .connect(&Secret::from(String::new()), &request(), limits())
            .await
            .is_err());
        let mut bad_limits = limits();
        bad_limits.max_frame_bytes = 0;
        assert!(service
            .connect(
                &Secret::from("per-call-key".to_owned()),
                &request(),
                bad_limits
            )
            .await
            .is_err());
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
    });
}

#[test]
fn websocket_schema_models_and_audio_formats_deserialize() {
    assert_eq!(
        MiniMaxStreamingTtsAudioFormat::Mp3,
        MiniMaxStreamingTtsAudioFormat::default()
    );
    assert_eq!(
        serde_json::from_value::<MiniMaxStreamingTtsModel>(json!("speech-2.6-hd")).unwrap(),
        MiniMaxStreamingTtsModel::Speech26Hd
    );
    assert_eq!(
        serde_json::from_value::<MiniMaxStreamingTtsAudioFormat>(json!("opus")).unwrap(),
        MiniMaxStreamingTtsAudioFormat::Opus
    );
}
