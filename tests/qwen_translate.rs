use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::mpsc,
    executor::block_on,
    stream::{self, BoxStream},
    StreamExt,
};
use lingxi_llm_client::providers::qwen::live_translate::Qwen35LiveTranslateConfig;
use lingxi_llm_client::providers::qwen::live_translate::Qwen35LiveTranslateInputAudioFormat;
use lingxi_llm_client::providers::qwen::live_translate::Qwen35LiveTranslateTurnDetection;
use lingxi_llm_client::providers::qwen::live_translate::Qwen38LiveTranslateConfig;
use lingxi_llm_client::providers::qwen::live_translate::Qwen38LiveTranslateTurnDetection;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateConfig;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateEvent;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateModel;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateOutputMode;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateRegion;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateRoute;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateSameLanguageSkip;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateScope;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateSession;
use lingxi_llm_client::providers::qwen::live_translate::QwenLiveTranslateVoiceCloneFrequency;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{
        RealtimeAudioFormat, RealtimeClose, RealtimeConnectRequest, RealtimeConnection,
        RealtimeError, RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSink,
        RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
}

struct FakePeer {
    incoming: mpsc::UnboundedSender<Incoming>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
}

fn fake_transport(preface: &[Value]) -> (Arc<FakeTransport>, FakePeer) {
    let (incoming_tx, incoming_rx) = mpsc::unbounded();
    let prefix = preface
        .iter()
        .map(|value| Ok(RealtimeFrame::text(value.to_string())))
        .collect::<Vec<_>>();
    let incoming = stream::iter(prefix).chain(incoming_rx).boxed();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let request = Arc::new(Mutex::new(None));
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(incoming)),
            sent: sent.clone(),
            request: request.clone(),
        }),
        FakePeer {
            incoming: incoming_tx,
            sent,
            request,
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
        let incoming = self
            .incoming
            .lock()
            .unwrap()
            .take()
            .expect("one connection");
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
            }),
            inbound: incoming,
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn ping(&mut self, _payload: bytes::Bytes) -> Result<(), RealtimeError> {
        Err(RealtimeError::InvalidInput {
            message: "test transport does not support explicit WebSocket Ping frames".into(),
        })
    }

    fn abort(&mut self) {
        // The test transport releases its local state when dropped.
    }

    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.sent.lock().unwrap().push(frame);
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}

fn route(region: QwenLiveTranslateRegion) -> QwenLiveTranslateRoute {
    QwenLiveTranslateRoute::new(region, "workspace-abc123").unwrap()
}

fn scope(route: &QwenLiveTranslateRoute) -> QwenLiveTranslateScope {
    QwenLiveTranslateScope::new("qwen-live", "account-9", route).unwrap()
}

fn limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 32,
        max_frame_bytes: 1024 * 1024,
    }
}

fn setup_events(model: &str) -> [Value; 2] {
    [
        json!({"type":"session.created","session":{"id":"sess-1","model":model}}),
        json!({"type":"session.updated","session":{"id":"sess-1","model":model}}),
    ]
}

fn sent_json(sent: &Arc<Mutex<Vec<RealtimeFrame>>>) -> Vec<Value> {
    sent.lock()
        .unwrap()
        .iter()
        .map(|frame| match frame {
            RealtimeFrame::Text(bytes) => serde_json::from_slice(bytes).unwrap(),
            RealtimeFrame::Binary(_) => panic!("LiveTranslate uses JSON text frames"),
        })
        .collect()
}

#[test]
fn route_scope_and_qwen35_session_update_are_regional_and_account_bound() {
    block_on(async {
        let beijing = route(QwenLiveTranslateRegion::Beijing);
        let singapore = route(QwenLiveTranslateRegion::Singapore);
        assert_eq!(
            beijing.endpoint(),
            "wss://workspace-abc123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
        );
        assert_eq!(
            singapore.endpoint(),
            "wss://workspace-abc123.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"
        );
        assert!(
            QwenLiveTranslateRoute::new(QwenLiveTranslateRegion::Beijing, "bad.workspace").is_err()
        );

        let (fake, peer) = fake_transport(&setup_events(
            "qwen3.5-livetranslate-flash-realtime-2026-05-19",
        ));
        let config = Qwen35LiveTranslateConfig {
            input_sample_rate_hz: 8_000,
            input_audio_format: Qwen35LiveTranslateInputAudioFormat::Opus,
            enable_source_transcription: true,
            source_language: Some("zh".into()),
            same_language_skip: Some(QwenLiveTranslateSameLanguageSkip {
                skip_text: false,
                skip_audio: true,
            }),
            hotwords: BTreeMap::from([("人工智能".into(), "Artificial Intelligence".into())]),
            voice: Some("default".into()),
            voice_clone_frequency: Some(QwenLiveTranslateVoiceCloneFrequency::Once),
            ..Default::default()
        };

        let (session, driver) = QwenLiveTranslateSession::connect(
            fake,
            beijing.clone(),
            scope(&beijing),
            Secret::from("ephemeral-key".to_owned()),
            QwenLiveTranslateConfig::qwen35(config),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let task = async move {
            assert!(matches!(
                events.next().await,
                Some(QwenLiveTranslateEvent::SessionCreated { .. })
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenLiveTranslateEvent::SessionUpdated { .. })
            ));
            assert_eq!(
                control.audio_format(),
                RealtimeAudioFormat::Encoded {
                    mime_type: "audio/opus".into()
                }
            );
            control
                .send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[1, 2, 3]),
                    format: RealtimeAudioFormat::Encoded {
                        mime_type: "audio/opus".into(),
                    },
                })
                .unwrap();
            control.close(RealtimeClose::normal("test")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(task, driver.run());
        driver_result.unwrap();

        let request = peer.request.lock().unwrap();
        let request = request.as_ref().unwrap();
        assert_eq!(request.endpoint, "wss://workspace-abc123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-livetranslate-flash-realtime");
        assert_eq!(
            request.headers,
            [("Authorization".into(), "Bearer ephemeral-key".into())]
        );
        assert!(!format!("{request:?}").contains("ephemeral-key"));

        let sent = sent_json(&peer.sent);
        let session = &sent[0]["session"];
        assert_eq!(session["modalities"], json!(["text", "audio"]));
        assert_eq!(session["sample_rate"], 8000);
        assert_eq!(session["input_audio_format"], "opus");
        assert_eq!(session["output_audio_format"], "pcm");
        assert_eq!(session["turn_detection"]["type"], "server_vad");
        assert_eq!(session["translation"]["language"], "en");
        assert_eq!(
            session["translation"]["same_language_skip_options"]["skip_audio"],
            true
        );
        assert_eq!(
            session["translation"]["corpus"]["phrases"]["人工智能"],
            "Artificial Intelligence"
        );
        assert_eq!(
            session["input_audio_transcription"]["model"],
            "qwen3-asr-flash-realtime"
        );
        assert_eq!(session["input_audio_transcription"]["language"], "zh");
        assert_eq!(session["voice"], "default");
        assert_eq!(session["enable_voice_clone"], true);
        assert_eq!(session["voice_clone_options"]["frequency"], "once");
        assert_eq!(
            sent[1],
            json!({"type":"input_audio_buffer.append","audio":"AQID"})
        );
    });
}

#[test]
fn qwen38_uses_its_nested_config_and_model_endpoint() {
    block_on(async {
        let route = route(QwenLiveTranslateRegion::Singapore);
        let (fake, peer) = fake_transport(&setup_events("qwen3.8-livetranslate-flash-realtime"));
        let config = Qwen38LiveTranslateConfig {
            output_mode: QwenLiveTranslateOutputMode::Text,
            target_language: "ja".into(),
            turn_detection: Qwen38LiveTranslateTurnDetection::ServerVad,
            voice: Some("Tina".into()),
            hotwords: BTreeMap::from([("AI".into(), "人工知能".into())]),
        };
        let (session, driver) = QwenLiveTranslateSession::connect(
            fake,
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenLiveTranslateConfig::qwen38(config),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                control.send(RealtimeInput::CommitAudio),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.close(RealtimeClose::normal("test")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(task, driver.run());
        driver_result.unwrap();

        assert_eq!(peer.request.lock().unwrap().as_ref().unwrap().endpoint, "wss://workspace-abc123.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3.8-livetranslate-flash-realtime");
        let sent = sent_json(&peer.sent);
        let session = &sent[0]["session"];
        assert!(session.get("modalities").is_none());
        assert_eq!(session["output_modalities"], json!(["text"]));
        assert_eq!(session["audio"]["input"]["format"]["sample_rate"], 16000);
        assert_eq!(session["audio"]["output"]["format"]["sample_rate"], 24000);
        assert_eq!(
            session["audio"]["input"]["turn_detection"]["type"],
            "server_vad"
        );
        assert_eq!(session["audio"]["output"]["voice"], "Tina");
        assert_eq!(session["translation"]["language"], "ja");
        assert_eq!(
            session["translation"]["corpus"]["phrases"]["AI"],
            "人工知能"
        );
    });
}

#[test]
fn controls_preserve_finish_final_events_and_require_explicit_close() {
    block_on(async {
        let route = route(QwenLiveTranslateRegion::Beijing);
        let (fake, peer) = fake_transport(&setup_events("qwen3.5-livetranslate-flash-realtime"));
        let config = Qwen35LiveTranslateConfig {
            turn_detection: Qwen35LiveTranslateTurnDetection::Manual,
            ..Default::default()
        };
        let (session, driver) = QwenLiveTranslateSession::connect(
            fake,
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenLiveTranslateConfig::qwen35(config),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let incoming = peer.incoming.clone();
        let task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(
                control.audio_format()
                    == RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000
                    }
            );
            assert!(matches!(
                control.send(RealtimeInput::Image {
                    data: Bytes::from_static(&[0xff, 0xd8, 0xff]),
                    mime_type: "image/jpeg".into()
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control
                .send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[1, 2, 3, 4]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000,
                    },
                })
                .unwrap();
            control
                .send(RealtimeInput::Image {
                    data: Bytes::from_static(&[0xff, 0xd8, 0xff, 0xe0]),
                    mime_type: "image/jpeg".into(),
                })
                .unwrap();
            let mut at_limit = vec![0xff, 0xd8, 0xff];
            at_limit.resize(500_000, 0);
            control
                .send(RealtimeInput::Image {
                    data: Bytes::from(at_limit),
                    mime_type: "image/jpeg".into(),
                })
                .unwrap();
            let mut over_limit = vec![0xff, 0xd8, 0xff];
            over_limit.resize(500_001, 0);
            assert!(matches!(
                control.send(RealtimeInput::Image {
                    data: Bytes::from(over_limit),
                    mime_type: "image/jpeg".into(),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.clear_audio().unwrap();
            assert!(matches!(
                control.commit_audio(),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control
                .send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[5, 6]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000,
                    },
                })
                .unwrap();
            control.commit_audio().unwrap();
            assert!(matches!(
                control.commit_audio(),
                Err(RealtimeError::InvalidInput { .. })
            ));
            assert!(matches!(
                control
                    .close_after_finished(RealtimeClose::normal("too early"))
                    .await,
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.finish().unwrap();
            for event in [
                json!({"type":"response.text.text","response_id":"r1","item_id":"i1","text":"confirmed ","stash":"tentative"}),
                json!({"type":"response.text.done","response_id":"r1","item_id":"i1","text":"final text"}),
                json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"i2","transcript":"source transcript","language":"zh","emotion":"neutral"}),
                json!({"type":"session.finished"}),
            ] {
                incoming
                    .unbounded_send(Ok(RealtimeFrame::text(event.to_string())))
                    .unwrap();
            }

            let mut saw_text = false;
            let mut saw_done = false;
            let mut saw_transcript = false;
            let mut saw_finished = false;
            for _ in 0..8 {
                match events.next().await {
                    Some(QwenLiveTranslateEvent::TextProgress35 { text, stash, .. }) => {
                        assert_eq!(text, "confirmed ");
                        assert_eq!(stash, "tentative");
                        saw_text = true;
                    }
                    Some(QwenLiveTranslateEvent::TextDone { text, .. }) => {
                        assert_eq!(text, "final text");
                        saw_done = true;
                    }
                    Some(QwenLiveTranslateEvent::SourceTranscriptionCompleted {
                        transcript,
                        ..
                    }) => {
                        assert_eq!(transcript, "source transcript");
                        saw_transcript = true;
                    }
                    Some(QwenLiveTranslateEvent::SessionFinished { .. }) => {
                        saw_finished = true;
                        break;
                    }
                    Some(_) => {}
                    None => break,
                }
            }
            assert!(saw_text && saw_done && saw_transcript && saw_finished);
            control
                .close_after_finished(RealtimeClose::normal("finished"))
                .await
                .unwrap();
        };
        let ((), driver_result) = futures::join!(task, driver.run());
        driver_result.unwrap();
        let sent = sent_json(&peer.sent);
        assert_eq!(
            sent[1],
            json!({"type":"input_audio_buffer.append","audio":"AQIDBA=="})
        );
        assert_eq!(
            sent[2],
            json!({"type":"input_image_buffer.append","image":"/9j/4A=="})
        );
        assert_eq!(sent[3]["type"], "input_image_buffer.append");
        assert_eq!(sent[3]["image"].as_str().unwrap().len(), 666_668);
        assert_eq!(sent[4], json!({"type":"input_audio_buffer.clear"}));
        assert_eq!(
            sent[5],
            json!({"type":"input_audio_buffer.append","audio":"BQY="})
        );
        assert_eq!(sent[6], json!({"type":"input_audio_buffer.commit"}));
        assert_eq!(sent[7], json!({"type":"session.finish"}));
    });
}

#[test]
fn model_specific_text_events_do_not_alias_qwen_omni_or_each_other() {
    block_on(async {
        let route = route(QwenLiveTranslateRegion::Beijing);
        let (fake, peer) = fake_transport(&setup_events("qwen3.8-livetranslate-flash-realtime"));
        for message in [
            json!({"type":"response.text.delta","response_id":"r","item_id":"i","delta":"text"}),
            json!({"type":"response.audio_transcript.delta","response_id":"r","item_id":"i","delta":"spoken"}),
            json!({"type":"conversation.item.input_audio_transcription.delta","item_id":"i","delta":"source"}),
            json!({"type":"response.audio.delta","response_id":"r","item_id":"i","delta":"AQID"}),
            json!({"type":"response.text.text","text":"must stay native for 3.8"}),
        ] {
            peer.incoming
                .unbounded_send(Ok(RealtimeFrame::text(message.to_string())))
                .unwrap();
        }
        let (session, driver) = QwenLiveTranslateSession::connect(
            fake,
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenLiveTranslateConfig::qwen38(Qwen38LiveTranslateConfig::default()),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(
                matches!(events.next().await, Some(QwenLiveTranslateEvent::TextDelta38 { delta, .. }) if delta == "text")
            );
            assert!(
                matches!(events.next().await, Some(QwenLiveTranslateEvent::AudioTranscriptDelta38 { delta, .. }) if delta == "spoken")
            );
            assert!(
                matches!(events.next().await, Some(QwenLiveTranslateEvent::SourceTranscriptionDelta38 { delta, .. }) if delta == "source")
            );
            assert!(
                matches!(events.next().await, Some(QwenLiveTranslateEvent::AudioDelta { data, format: RealtimeAudioFormat::Pcm16 { sample_rate_hz: 24_000 }, .. }) if data == Bytes::from_static(&[1, 2, 3]))
            );
            assert!(
                matches!(events.next().await, Some(QwenLiveTranslateEvent::Native { event_type, .. }) if event_type == "response.text.text")
            );
            control.close(RealtimeClose::normal("test")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(task, driver.run());
        driver_result.unwrap();
    });
}

#[test]
fn rejects_mismatched_models_invalid_vad_and_setup_errors_before_use() {
    block_on(async {
        assert!(Qwen35LiveTranslateConfig::new(
            QwenLiveTranslateModel::Qwen38LiveTranslateFlashRealtime
        )
        .is_err());
        let route = route(QwenLiveTranslateRegion::Beijing);
        let (fake, peer) = fake_transport(&setup_events("qwen3.5-livetranslate-flash-realtime"));
        let config = Qwen35LiveTranslateConfig {
            turn_detection: Qwen35LiveTranslateTurnDetection::ServerVad {
                threshold: 1.5,
                silence_duration_ms: 1000,
            },
            ..Default::default()
        };
        let result = QwenLiveTranslateSession::connect(
            fake,
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenLiveTranslateConfig::qwen35(config),
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
        assert!(peer.request.lock().unwrap().is_none());

        let error_preface = [json!({"type":"error","error":{"code":"invalid_value"}})];
        let (fake, peer) = fake_transport(&error_preface);
        let result = QwenLiveTranslateSession::connect(
            fake,
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenLiveTranslateConfig::default(),
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::Transport { .. })));
        assert!(peer.sent.lock().unwrap().is_empty());

        let mismatched_preface = [json!({
            "type":"session.created",
            "session":{"id":"sess-1","model":"qwen3.8-livetranslate-flash-realtime"}
        })];
        let (fake, peer) = fake_transport(&mismatched_preface);
        let result = QwenLiveTranslateSession::connect(
            fake,
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenLiveTranslateConfig::default(),
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::Transport { .. })));
        assert!(peer.sent.lock().unwrap().is_empty());
    });
}
