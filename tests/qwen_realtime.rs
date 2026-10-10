use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::mpsc,
    executor::block_on,
    stream::{self, BoxStream},
    StreamExt,
};
use lingxi_llm_client::providers::google::live::GeminiLiveConfig;
use lingxi_llm_client::providers::google::live::GeminiLiveSession;
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeCodec;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeConfig;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeEvent;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeModel;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeOutputMode;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeRegion;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeRoute;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeScope;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeSession;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeTurnDetection;
use lingxi_llm_client::providers::qwen::realtime::QwenRealtimeVideoMode;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeConfig;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeSession;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeConfig;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeRoute;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeScope;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeSession;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{
        RealtimeAudioFormat, RealtimeClose, RealtimeCodec, RealtimeConnectRequest,
        RealtimeConnection, RealtimeError, RealtimeEvent, RealtimeFrame, RealtimeInput,
        RealtimeLimits, RealtimeSink, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
}

struct FakePeer {
    incoming: mpsc::UnboundedSender<Incoming>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
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
    let connect_count = Arc::new(Mutex::new(0));
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(incoming)),
            sent: sent.clone(),
            request: request.clone(),
            connect_count: connect_count.clone(),
        }),
        FakePeer {
            incoming: incoming_tx,
            sent,
            request,
            connect_count,
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

fn transport(transport: Arc<FakeTransport>) -> Arc<dyn RealtimeTransport> {
    transport
}

fn route() -> QwenRealtimeRoute {
    QwenRealtimeRoute::new(QwenRealtimeRegion::Beijing, "workspace-abc123").unwrap()
}

fn scope(route: &QwenRealtimeRoute) -> QwenRealtimeScope {
    QwenRealtimeScope::new("qwen-mainland", "billing-account-7", route).unwrap()
}

fn limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 24,
        max_frame_bytes: 512 * 1024,
    }
}

fn setup_events() -> [Value; 2] {
    [
        json!({"type":"session.created","session":{"id":"sess-1","modalities":["text","audio"]}}),
        json!({"type":"session.updated","session":{"modalities":["text","audio"]}}),
    ]
}

fn sent_json(sent: &Arc<Mutex<Vec<RealtimeFrame>>>) -> Vec<Value> {
    sent.lock()
        .unwrap()
        .iter()
        .map(|frame| match frame {
            RealtimeFrame::Text(bytes) => serde_json::from_slice(bytes).unwrap(),
            RealtimeFrame::Binary(_) => panic!("Qwen Realtime events use JSON text frames"),
        })
        .collect()
}

fn connect_request(endpoint: &str) -> RealtimeConnectRequest {
    RealtimeConnectRequest {
        endpoint: endpoint.into(),
        headers: vec![("Authorization".into(), "Bearer hidden".into())],
        max_frame_bytes: 1,
    }
}

#[test]
fn regional_routes_scope_workspace_and_per_call_auth_are_bound() {
    block_on(async {
        let beijing =
            QwenRealtimeRoute::new(QwenRealtimeRegion::Beijing, "workspace-abc123").unwrap();
        let singapore =
            QwenRealtimeRoute::new(QwenRealtimeRegion::Singapore, "workspace-abc123").unwrap();
        assert_eq!(
            beijing.endpoint(),
            "wss://workspace-abc123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
        );
        assert_eq!(
            singapore.endpoint(),
            "wss://workspace-abc123.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"
        );
        assert!(QwenRealtimeRoute::new(QwenRealtimeRegion::Beijing, "bad.workspace").is_err());
        assert!(QwenRealtimeRoute::new(QwenRealtimeRegion::Beijing, "-bad-workspace").is_err());

        let (fake, peer) = fake_transport(&setup_events());
        let beijing_scope = scope(&beijing);
        let result = QwenRealtimeSession::connect(
            transport(fake.clone()),
            singapore,
            beijing_scope,
            Secret::from("ephemeral-key".to_owned()),
            QwenRealtimeConfig::default(),
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);

        let route = route();
        let scope = scope(&route);
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = QwenRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope.clone(),
            Secret::from("ephemeral-key".to_owned()),
            QwenRealtimeConfig {
                instructions: Some("Be concise.".into()),
                video_mode: Some(QwenRealtimeVideoMode::Compact),
                ..QwenRealtimeConfig::default()
            },
            limits(),
        )
        .await
        .unwrap();
        assert_eq!(session.scope(), &scope);
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::SessionCreated { .. })
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::SessionUpdated { .. })
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let request = peer.request.lock().unwrap();
        let request = request.as_ref().unwrap();
        assert_eq!(
            request.endpoint,
            "wss://workspace-abc123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3.8-omni-flash-realtime"
        );
        assert_eq!(
            request.headers,
            [("Authorization".into(), "Bearer ephemeral-key".into())]
        );
        assert!(!format!("{request:?}").contains("ephemeral-key"));

        let sent = sent_json(&peer.sent);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0]["type"], "session.update");
        assert_eq!(sent[0]["session"]["model"], "qwen3.8-omni-flash-realtime");
        assert_eq!(sent[0]["session"]["modalities"], json!(["text", "audio"]));
        assert_eq!(sent[0]["session"]["turn_detection"]["type"], "server_vad");
        assert_eq!(sent[0]["session"]["enable_input_audio_transcription"], true);
        assert_eq!(sent[0]["session"]["instructions"], "Be concise.");
        assert_eq!(
            sent[0]["session"]["video"]["input"]["representation_compact"],
            "normal"
        );
    });
}

#[test]
fn clear_audio_uses_qwen_native_input_buffer_event() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = QwenRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            control.send(RealtimeInput::ClearAudio).unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[1], json!({"type":"input_audio_buffer.clear"}));
    });
}

#[test]
fn manual_audio_image_text_and_cancel_use_qwen_wire_events() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let mut config = QwenRealtimeConfig::new(QwenRealtimeModel::Qwen35OmniPlusRealtime);
        config.turn_detection = QwenRealtimeTurnDetection::Manual;
        config.output_mode = QwenRealtimeOutputMode::Text;
        config.enable_input_audio_transcription = false;
        let (session, driver) = QwenRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            config,
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
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
            control.send(RealtimeInput::CommitAudio).unwrap();
            control.send(RealtimeInput::ContinueResponse).unwrap();
            control.send(RealtimeInput::Text("hello".into())).unwrap();
            control.interrupt().unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(sent[0]["session"]["modalities"], json!(["text"]));
        assert_eq!(sent[0]["session"]["turn_detection"], Value::Null);
        assert_eq!(
            sent[0]["session"]["enable_input_audio_transcription"],
            false
        );
        assert_eq!(
            sent[1],
            json!({"type":"input_audio_buffer.append","audio":"AQIDBA=="})
        );
        assert_eq!(
            sent[2],
            json!({"type":"input_image_buffer.append","image":"/9j/4A=="})
        );
        assert_eq!(sent[3], json!({"type":"input_audio_buffer.commit"}));
        assert_eq!(sent[4], json!({"type":"response.create"}));
        assert_eq!(sent[5]["type"], "conversation.item.create");
        assert_eq!(sent[5]["item"]["content"][0]["type"], "input_text");
        assert_eq!(sent[5]["item"]["content"][0]["text"], "hello");
        assert_eq!(sent[6], json!({"type":"response.create"}));
        assert_eq!(sent[7], json!({"type":"response.cancel"}));
    });
}

#[test]
fn vad_mode_rejects_manual_boundaries_and_image_frames_require_audio_first() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = QwenRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let image = RealtimeInput::Image {
                data: Bytes::from_static(&[0xff, 0xd8, 0xff]),
                mime_type: "image/jpeg".into(),
            };
            assert!(matches!(
                control.send(image),
                Err(RealtimeError::InvalidInput { .. })
            ));
            assert!(matches!(
                control.send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[1, 2]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 24_000,
                    },
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control
                .send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[1, 2]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000,
                    },
                })
                .unwrap();
            assert!(matches!(
                control.send(RealtimeInput::Image {
                    data: Bytes::from_static(&[1, 2]),
                    mime_type: "image/png".into(),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            let oversized = Bytes::from(vec![0_u8; 196_609]);
            assert!(matches!(
                control.send(RealtimeInput::Image {
                    data: oversized,
                    mime_type: "image/jpeg".into(),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            assert!(matches!(
                control.send(RealtimeInput::CommitAudio),
                Err(RealtimeError::InvalidInput { .. })
            ));
            assert!(matches!(
                control.send(RealtimeInput::ContinueResponse),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.interrupt().unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        let sent = sent_json(&peer.sent);
        assert_eq!(sent.len(), 3);
        assert_eq!(sent[1]["type"], "input_audio_buffer.append");
        assert_eq!(sent[2]["type"], "response.cancel");
    });
}

#[test]
fn output_audio_text_transcript_usage_and_native_errors_are_typed() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = QwenRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        for message in [
            json!({"type":"response.audio.delta","response_id":"resp-1","item_id":"item-1","output_index":0,"content_index":0,"delta":"AQIDBA=="}),
            json!({"type":"response.audio_transcript.delta","response_id":"resp-1","item_id":"item-1","delta":"hi"}),
            json!({"type":"response.audio_transcript.done","response_id":"resp-1","item_id":"item-1","transcript":"hello"}),
            json!({"type":"response.text.delta","response_id":"resp-2","item_id":"item-2","delta":"text"}),
            json!({"type":"response.text.done","response_id":"resp-2","item_id":"item-2","text":"text done"}),
            json!({"type":"response.done","response":{"id":"resp-1","status":"completed","usage":{"input_tokens":3,"output_tokens":4}}}),
            json!({"type":"error","error":{"type":"invalid_request_error","code":"invalid_value","message":"unsupported setting","param":"session.modalities"}}),
        ] {
            peer.incoming
                .unbounded_send(Ok(RealtimeFrame::text(message.to_string())))
                .unwrap();
        }
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::AudioDelta {
                    response_id: Some(ref response_id),
                    data,
                    format: RealtimeAudioFormat::Pcm16 { sample_rate_hz: 24_000 },
                    ..
                }) if response_id == "resp-1" && data == Bytes::from_static(&[1, 2, 3, 4])
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::AudioTranscriptDelta { ref delta, .. }) if delta == "hi"
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::AudioTranscriptDone { ref transcript, .. }) if transcript == "hello"
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::TextDelta { ref delta, .. }) if delta == "text"
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::TextDone { ref text, .. }) if text == "text done"
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::ResponseDone { response, .. })
                    if response["usage"]["output_tokens"] == 4
            ));
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::ProviderError {
                    error_type: Some(ref error_type),
                    code: Some(ref code),
                    param: Some(ref param),
                    ..
                }) if error_type == "invalid_request_error" && code == "invalid_value" && param == "session.modalities"
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
    });
}

#[test]
fn unexpected_disconnect_and_invalid_audio_delta_are_reported_without_replay() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = QwenRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming.close_channel();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::Realtime(
                    RealtimeEvent::ConnectionInterrupted { .. }
                ))
            ));
            drop(control);
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        assert!(matches!(
            driver_result,
            Err(RealtimeError::UnexpectedRemoteClose)
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);

        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = QwenRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("key".to_owned()),
            QwenRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"type":"response.audio.delta","delta":"not base64!"}"#,
            )))
            .unwrap();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(QwenRealtimeEvent::Realtime(RealtimeEvent::ProviderError {
                    code: Some(ref code),
                    ..
                })) if code == "invalid_provider_frame"
            ));
            drop(control);
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        assert!(matches!(driver_result, Err(RealtimeError::Codec { .. })));
    });
}

#[test]
fn image_input_support_is_provider_specific() {
    block_on(async {
        let image = RealtimeInput::Image {
            data: Bytes::from_static(&[0xff, 0xd8, 0xff]),
            mime_type: "image/jpeg".into(),
        };
        assert!(OpenAiRealtimeCodec::default().encode(&image).is_ok());

        let (fake, _) = fake_transport(&[json!({"setupComplete":{}})]);
        let (session, _driver) = GeminiLiveSession::connect(
            transport(fake),
            connect_request("wss://generativelanguage.googleapis.com/ws/live?key=hidden"),
            GeminiLiveConfig::new("gemini-live-test", "google-account"),
            limits(),
        )
        .await
        .unwrap();
        let (control, _events) = session.into_parts();
        assert!(control.send(image.clone()).is_ok());

        let (fake, _) = fake_transport(&[
            json!({"type":"session.created","session":{"id":"glm-session"}}),
            json!({"type":"session.updated","session":{}}),
        ]);
        let route = GlmRealtimeRoute::mainland_china();
        let (session, _driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            GlmRealtimeScope::new("glm", "glm-account", &route).unwrap(),
            Secret::from("glm-key".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, _events) = session.into_parts();
        assert!(matches!(
            control.send(image.clone()),
            Err(RealtimeError::InvalidInput { .. })
        ));

        let (fake, _) = fake_transport(&[
            json!({"type":"session.created","session":{"id":"xai-session"}}),
            json!({"type":"session.updated","session":{}}),
        ]);
        let (session, _driver) = XaiRealtimeSession::connect(
            transport(fake),
            connect_request("wss://api.x.ai/v1/realtime?model=grok-voice-latest"),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, _events) = session.into_parts();
        assert!(matches!(
            control.send(image),
            Err(RealtimeError::InvalidInput { .. })
        ));
    });
}
