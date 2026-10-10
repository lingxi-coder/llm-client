use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::{mpsc, oneshot},
    executor::block_on,
    stream::{self, BoxStream},
    StreamExt,
};
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeAudioTransport;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeCommand;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeConfig;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeEvent;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeFunctionTool;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeReasoningEffort;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeResumeRef;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeSession;
use lingxi_llm_client::providers::xai::realtime::XaiRealtimeVadOptions;
use lingxi_llm_client::providers::xai::realtime::XaiTurnDetection;
use lingxi_llm_client::realtime::{
    RealtimeAudioFormat, RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError,
    RealtimeEvent, RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSink, RealtimeToolResult,
    RealtimeTransport, MAX_REALTIME_TOOL_RESULTS,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    expected_endpoint: String,
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    sent_notifications: mpsc::UnboundedSender<RealtimeFrame>,
    connect_count: Arc<Mutex<usize>>,
}

struct FakePeer {
    incoming: mpsc::UnboundedSender<Incoming>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    sent_notifications: mpsc::UnboundedReceiver<RealtimeFrame>,
    connect_count: Arc<Mutex<usize>>,
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    sent_notifications: mpsc::UnboundedSender<RealtimeFrame>,
}

fn fake_transport(preface: &[&str]) -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_at(
        preface,
        "wss://api.x.ai/v1/realtime?model=grok-voice-latest",
    )
}

fn fake_transport_at(preface: &[&str], expected_endpoint: &str) -> (Arc<FakeTransport>, FakePeer) {
    let (incoming_tx, incoming_rx) = mpsc::unbounded();
    let (sent_notifications, sent_notification_rx) = mpsc::unbounded();
    let prefix = preface
        .iter()
        .map(|message| Ok(RealtimeFrame::text((*message).to_owned())))
        .collect::<Vec<_>>();
    let incoming = stream::iter(prefix).chain(incoming_rx).boxed();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let connect_count = Arc::new(Mutex::new(0));
    (
        Arc::new(FakeTransport {
            expected_endpoint: expected_endpoint.to_owned(),
            incoming: Mutex::new(Some(incoming)),
            sent: sent.clone(),
            sent_notifications: sent_notifications.clone(),
            connect_count: connect_count.clone(),
        }),
        FakePeer {
            incoming: incoming_tx,
            sent,
            sent_notifications: sent_notification_rx,
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
        assert_eq!(request.endpoint, self.expected_endpoint);
        assert_eq!(request.max_frame_bytes, 16 * 1024);
        *self.connect_count.lock().unwrap() += 1;
        let incoming = self
            .incoming
            .lock()
            .unwrap()
            .take()
            .expect("one session per fake transport");
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
                sent_notifications: self.sent_notifications.clone(),
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
        self.sent.lock().unwrap().push(frame.clone());
        let _ = self.sent_notifications.unbounded_send(frame);
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}

fn request() -> RealtimeConnectRequest {
    RealtimeConnectRequest {
        endpoint: "wss://api.x.ai/v1/realtime?model=grok-voice-latest".into(),
        headers: vec![("Authorization".into(), "Bearer hidden-token".into())],
        max_frame_bytes: 1,
    }
}

fn limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 16,
        max_frame_bytes: 16 * 1024,
    }
}

fn as_transport(transport: Arc<FakeTransport>) -> Arc<dyn RealtimeTransport> {
    transport
}

fn sent_json(sent: &Arc<Mutex<Vec<RealtimeFrame>>>) -> Vec<Value> {
    sent.lock()
        .unwrap()
        .iter()
        .filter_map(|frame| match frame {
            RealtimeFrame::Text(bytes) => serde_json::from_slice(bytes).ok(),
            RealtimeFrame::Binary(_) => None,
        })
        .collect()
}

fn sent_frame_json(frame: RealtimeFrame) -> Value {
    match frame {
        RealtimeFrame::Text(bytes) => serde_json::from_slice(&bytes).unwrap(),
        RealtimeFrame::Binary(_) => panic!("frame is binary"),
    }
}

const CREATED_PREFACE: [&str; 3] = [
    r#"{"type":"session.created","session":{"id":"sess-1"}}"#,
    r#"{"type":"conversation.created","conversation":{"id":"conv-1"}}"#,
    r#"{"type":"session.updated","session":{"voice":"eve"}}"#,
];

#[test]
fn setup_waits_for_session_created_and_session_updated() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        let config = XaiRealtimeConfig {
            instructions: Some("Answer briefly.".into()),
            ..Default::default()
        };
        let (session, driver) =
            XaiRealtimeSession::connect(as_transport(transport), request(), config, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();

        let session_task = async move {
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::SessionCreated { .. })
            ));
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::ConversationCreated { .. })
            ));
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::SessionUpdated { .. })
            ));
            control.send(RealtimeInput::Text("Hello".into())).unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
            assert_eq!(
                events.next().await,
                Some(XaiRealtimeEvent::Realtime(RealtimeEvent::Closed {
                    code: 1000,
                    reason: "done".into(),
                }))
            );
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(sent[0]["type"], "session.update");
        assert_eq!(sent[0]["session"]["voice"], "eve");
        assert_eq!(sent[0]["session"]["instructions"], "Answer briefly.");
        assert_eq!(sent[0]["session"]["turn_detection"]["type"], "server_vad");
        assert_eq!(sent[0]["session"]["audio"]["input"]["transport"], "json");
        assert_eq!(
            sent[0]["session"]["audio"]["input"]["format"]["type"],
            "audio/pcm"
        );
        assert_eq!(
            sent[0]["session"]["audio"]["input"]["format"]["rate"],
            24_000
        );
        assert_eq!(sent[1]["type"], "conversation.item.create");
        assert_eq!(sent[1]["item"]["content"][0]["text"], "Hello");
        assert_eq!(sent[2], json!({ "type": "response.create" }));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
    });
}

#[test]
fn session_settings_encode_documented_controls_and_function_tools() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        let mut replace = std::collections::BTreeMap::new();
        replace.insert("Acme Mobile".to_owned(), "Acme Mobull".to_owned());
        let config = XaiRealtimeConfig {
            reasoning_effort: Some(XaiRealtimeReasoningEffort::None),
            vad: XaiRealtimeVadOptions {
                threshold: Some(0.7),
                silence_duration_ms: Some(640),
                prefix_padding_ms: Some(333),
                idle_timeout_ms: Some(30_000),
            },
            input_transcription: true,
            input_transcription_language_hint: Some("ja".into()),
            input_transcription_keyterms: vec!["Grok".into(), "xAI".into()],
            output_audio_speed: Some(1.2),
            replace,
            tools: vec![XaiRealtimeFunctionTool::new(
                "lookup",
                json!({
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"]
                }),
            )
            .with_description("Look up a value")],
            resumption_enabled: true,
            credential_scope: "customer-account-1".into(),
            ..Default::default()
        };
        let (session, driver) =
            XaiRealtimeSession::connect(as_transport(transport), request(), config, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _created = events.next().await;
            let _conversation = events.next().await;
            let _updated = events.next().await;
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let update = sent_json(&peer.sent).remove(0);
        assert_eq!(update["session"]["reasoning"]["effort"], "none");
        assert_eq!(update["session"]["turn_detection"]["threshold"], 0.7);
        assert_eq!(
            update["session"]["turn_detection"]["silence_duration_ms"],
            640
        );
        assert_eq!(
            update["session"]["turn_detection"]["prefix_padding_ms"],
            333
        );
        assert_eq!(
            update["session"]["turn_detection"]["idle_timeout_ms"],
            30_000
        );
        assert_eq!(
            update["session"]["audio"]["input"]["transcription"]["model"],
            "grok-transcribe"
        );
        assert_eq!(
            update["session"]["audio"]["input"]["transcription"]["language_hint"],
            "ja"
        );
        assert_eq!(
            update["session"]["audio"]["input"]["transcription"]["keyterms"],
            json!(["Grok", "xAI"])
        );
        assert_eq!(update["session"]["audio"]["output"]["speed"], 1.2);
        assert_eq!(update["session"]["replace"]["Acme Mobile"], "Acme Mobull");
        assert_eq!(update["session"]["resumption"]["enabled"], true);
        assert_eq!(
            update["session"]["tools"][0],
            json!({
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"]
                }
            })
        );
    });
}

#[test]
fn invalid_documented_settings_fail_before_transport_connect() {
    block_on(async {
        for config in [
            XaiRealtimeConfig {
                vad: XaiRealtimeVadOptions {
                    threshold: Some(0.91),
                    ..Default::default()
                },
                ..Default::default()
            },
            XaiRealtimeConfig {
                input_transcription: true,
                input_transcription_keyterms: vec!["x".repeat(51)],
                ..Default::default()
            },
            XaiRealtimeConfig {
                output_audio_speed: Some(1.6),
                ..Default::default()
            },
            XaiRealtimeConfig {
                turn_detection: XaiTurnDetection::Manual,
                vad: XaiRealtimeVadOptions {
                    silence_duration_ms: Some(100),
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            let (transport, peer) = fake_transport(&CREATED_PREFACE);
            assert!(XaiRealtimeSession::connect(
                as_transport(transport),
                request(),
                config,
                limits()
            )
            .await
            .is_err());
            assert_eq!(*peer.connect_count.lock().unwrap(), 0);
        }
    });
}

#[test]
fn provider_commands_clear_audio_force_message_and_override_one_response() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        let (session, driver) = XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _created = events.next().await;
            let _conversation = events.next().await;
            let _updated = events.next().await;
            control
                .send_command(XaiRealtimeCommand::ClearAudioBuffer)
                .unwrap();
            control
                .send_command(XaiRealtimeCommand::ForceMessage {
                    text: "This is a notice.".into(),
                    interruptible: Some(false),
                })
                .unwrap();
            control
                .send_command(XaiRealtimeCommand::CreateResponse {
                    instructions: Some("Answer in Spanish for this response.".into()),
                })
                .unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        let sent = sent_json(&peer.sent);
        assert_eq!(sent[1], json!({ "type": "input_audio_buffer.clear" }));
        assert_eq!(
            sent[2],
            json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "force_message",
                    "role": "assistant",
                    "interruptible": false,
                    "content": [{ "type": "output_text", "text": "This is a notice." }]
                }
            })
        );
        assert_eq!(
            sent[3],
            json!({
                "type": "response.create",
                "response": {
                    "instructions": "Answer in Spanish for this response."
                }
            })
        );
    });
}

#[test]
fn binary_transport_accepts_raw_opus_and_types_binary_output() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::binary(Bytes::from_static(&[9, 8, 7]))))
            .unwrap();
        let format = RealtimeAudioFormat::Encoded {
            mime_type: "audio/opus".into(),
        };
        let config = XaiRealtimeConfig {
            input_audio_format: format.clone(),
            output_audio_format: format.clone(),
            input_audio_transport: XaiRealtimeAudioTransport::Binary,
            output_audio_transport: XaiRealtimeAudioTransport::Binary,
            ..Default::default()
        };
        let (session, driver) =
            XaiRealtimeSession::connect(as_transport(transport), request(), config, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _created = events.next().await;
            let _conversation = events.next().await;
            let _updated = events.next().await;
            control
                .send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[1, 2, 3]),
                    format: format.clone(),
                })
                .unwrap();
            assert_eq!(
                events.next().await,
                Some(XaiRealtimeEvent::AudioBinaryDelta {
                    data: Bytes::from_static(&[9, 8, 7]),
                    format,
                })
            );
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = peer.sent.lock().unwrap();
        let update: Value = serde_json::from_slice(match &sent[0] {
            RealtimeFrame::Text(bytes) => bytes,
            RealtimeFrame::Binary(_) => panic!("session.update remains JSON"),
        })
        .unwrap();
        assert_eq!(
            update["session"]["audio"]["input"]["format"]["type"],
            "audio/opus"
        );
        assert_eq!(update["session"]["audio"]["input"]["transport"], "binary");
        assert_eq!(update["session"]["audio"]["output"]["transport"], "binary");
        assert_eq!(
            sent[1],
            RealtimeFrame::binary(Bytes::from_static(&[1, 2, 3]))
        );
    });
}

#[test]
fn resumption_adds_conversation_id_as_a_query_parameter() {
    block_on(async {
        let endpoint = "wss://api.x.ai/v1/realtime?model=grok-voice-latest&conversation_id=conv-2";
        let resume_ref = XaiRealtimeResumeRef::import(
            "conv-2",
            "grok-voice-latest",
            "wss://api.x.ai/v1/realtime?model=grok-voice-latest",
            "customer-account-1",
        )
        .unwrap();
        let resume_debug = format!("{resume_ref:?}");
        assert!(!resume_debug.contains("conv-2"));
        assert!(!resume_debug.contains("customer-account-1"));
        assert!(!resume_debug.contains("wss://"));
        for mismatched_config in [
            XaiRealtimeConfig {
                resumption_enabled: true,
                credential_scope: "customer-account-2".into(),
                resume_ref: Some(resume_ref.clone()),
                ..Default::default()
            },
            XaiRealtimeConfig {
                model: "grok-voice-think-fast-2.0".into(),
                resumption_enabled: true,
                credential_scope: "customer-account-1".into(),
                resume_ref: Some(resume_ref.clone()),
                ..Default::default()
            },
        ] {
            let (transport, peer) = fake_transport(&CREATED_PREFACE);
            assert!(XaiRealtimeSession::connect(
                as_transport(transport),
                request(),
                mismatched_config,
                limits()
            )
            .await
            .is_err());
            assert_eq!(*peer.connect_count.lock().unwrap(), 0);
        }
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        assert!(XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig {
                model: "grok-voice-think-fast-2.0".into(),
                ..Default::default()
            },
            limits()
        )
        .await
        .is_err());
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);

        let wrong_endpoint = "wss://api.x.ai/v1/another-realtime?model=grok-voice-latest";
        let (transport, peer) = fake_transport_at(&CREATED_PREFACE, wrong_endpoint);
        let error = XaiRealtimeSession::connect(
            as_transport(transport),
            RealtimeConnectRequest {
                endpoint: wrong_endpoint.into(),
                ..request()
            },
            XaiRealtimeConfig {
                resumption_enabled: true,
                credential_scope: "customer-account-1".into(),
                resume_ref: Some(resume_ref.clone()),
                ..Default::default()
            },
            limits(),
        )
        .await;
        assert!(error.is_err());
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);

        let (transport, peer) = fake_transport_at(&CREATED_PREFACE, endpoint);
        let config = XaiRealtimeConfig {
            resumption_enabled: true,
            credential_scope: "customer-account-1".into(),
            resume_ref: Some(resume_ref),
            ..Default::default()
        };
        let (session, driver) =
            XaiRealtimeSession::connect(as_transport(transport), request(), config, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _created = events.next().await;
            let _conversation = events.next().await;
            let _updated = events.next().await;
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        assert_eq!(
            sent_json(&peer.sent)[0]["session"]["resumption"]["enabled"],
            true
        );
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
    });
}

#[test]
fn newly_supported_lifecycle_events_are_typed_and_keep_native_payloads() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        for message in [
            r#"{"type":"input_audio_buffer.cleared","event_id":"evt-clear"}"#,
            r#"{"type":"input_audio_buffer.timeout_triggered","event_id":"evt-timeout"}"#,
            r#"{"type":"conversation.item.added","item":{"id":"item-1","type":"message"}}"#,
            r#"{"type":"conversation.item.deleted","item_id":"item-1"}"#,
            r#"{"type":"conversation.item.truncated","item_id":"item-1"}"#,
            r#"{"type":"response.output_item.added","item":{"id":"item-2"}}"#,
            r#"{"type":"response.output_item.done","item":{"id":"item-2"}}"#,
            r#"{"type":"response.content_part.added","part":{"type":"text"}}"#,
            r#"{"type":"response.content_part.done","part":{"type":"text"}}"#,
            r#"{"type":"mcp_list_tools.completed","tools":[]}"#,
        ] {
            peer.incoming
                .unbounded_send(Ok(RealtimeFrame::text(message)))
                .unwrap();
        }
        let (session, driver) = XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _created = events.next().await;
            let _conversation = events.next().await;
            let _updated = events.next().await;
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::InputAudioBufferCleared { native }) if native["event_id"] == "evt-clear")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::InputAudioBufferTimeoutTriggered { native }) if native["event_id"] == "evt-timeout")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::ConversationItemAdded { item_id: Some(id), .. }) if id == "item-1")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::ConversationItemDeleted { item_id: Some(id), .. }) if id == "item-1")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::ConversationItemTruncated { item_id: Some(id), .. }) if id == "item-1")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::ResponseOutputItemAdded { native }) if native["item"]["id"] == "item-2")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::ResponseOutputItemDone { native }) if native["item"]["id"] == "item-2")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::ResponseContentPartAdded { native }) if native["part"]["type"] == "text")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::ResponseContentPartDone { native }) if native["part"]["type"] == "text")
            );
            assert!(
                matches!(events.next().await, Some(XaiRealtimeEvent::McpEvent { event_type, native }) if event_type == "mcp_list_tools.completed" && native["tools"].is_array())
            );
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
    });
}

#[test]
fn audio_input_is_base64_and_audio_output_is_a_typed_native_event() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        let config = XaiRealtimeConfig {
            input_audio_format: RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: 16_000,
            },
            output_audio_format: RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: 16_000,
            },
            ..Default::default()
        };
        let (session, driver) =
            XaiRealtimeSession::connect(as_transport(transport), request(), config, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"type":"response.output_audio.delta","response_id":"resp-1","item_id":"item-2","output_index":0,"content_index":1,"delta":"AQIDBA=="}"#,
            )))
            .unwrap();

        let session_task = async move {
            let _session_created = events.next().await;
            let _conversation_created = events.next().await;
            let _session_updated = events.next().await;
            control
                .send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[1, 2, 3]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000,
                    },
                })
                .unwrap();
            assert_eq!(
                events.next().await,
                Some(XaiRealtimeEvent::AudioDelta {
                    response_id: Some("resp-1".into()),
                    item_id: Some("item-2".into()),
                    output_index: Some(0),
                    content_index: Some(1),
                    data: Bytes::from_static(&[1, 2, 3, 4]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000,
                    },
                    native: json!({
                        "type": "response.output_audio.delta",
                        "response_id": "resp-1",
                        "item_id": "item-2",
                        "output_index": 0,
                        "content_index": 1,
                        "delta": "AQIDBA=="
                    }),
                })
            );
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(
            sent[0]["session"]["audio"]["input"]["format"]["rate"],
            16_000
        );
        assert_eq!(
            sent[1],
            json!({
                "type": "input_audio_buffer.append",
                "audio": "AQID"
            })
        );
    });
}

#[test]
fn manual_turn_detection_supports_commit_and_cancel_but_vad_rejects_them() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        let config = XaiRealtimeConfig {
            turn_detection: XaiTurnDetection::Manual,
            ..Default::default()
        };
        let (session, driver) =
            XaiRealtimeSession::connect(as_transport(transport), request(), config, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ready = events.next().await;
            let _ready = events.next().await;
            let _ready = events.next().await;
            control.send(RealtimeInput::CommitAudio).unwrap();
            control.interrupt().unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        let sent = sent_json(&peer.sent);
        assert_eq!(sent[0]["session"]["turn_detection"]["type"], Value::Null);
        assert_eq!(sent[1], json!({ "type": "input_audio_buffer.commit" }));
        assert_eq!(sent[2], json!({ "type": "response.cancel" }));

        let (vad_transport, _) = fake_transport(&CREATED_PREFACE);
        let (session, _driver) = XaiRealtimeSession::connect(
            as_transport(vad_transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, _events) = session.into_parts();
        assert!(matches!(
            control.send(RealtimeInput::CommitAudio),
            Err(RealtimeError::InvalidInput { .. })
        ));
        assert!(matches!(
            control.interrupt(),
            Err(RealtimeError::InvalidInput { .. })
        ));
    });
}

#[test]
fn transcription_response_and_speech_events_are_typed() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        let config = XaiRealtimeConfig {
            input_transcription: true,
            ..Default::default()
        };
        let (session, driver) =
            XaiRealtimeSession::connect(as_transport(transport), request(), config, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        for message in [
            r#"{"type":"input_audio_buffer.speech_started","item_id":"user-a","audio_start_ms":120}"#,
            r#"{"type":"conversation.item.input_audio_transcription.updated","item_id":"user-a","transcript":"hel"}"#,
            r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"user-a","transcript":"hello"}"#,
            r#"{"type":"response.text.delta","response_id":"resp-a","item_id":"asst-a","delta":"hi"}"#,
            r#"{"type":"response.function_call_arguments.done","response_id":"resp-a","item_id":"tool-a","call_id":"call-a","name":"lookup","arguments":"{\"q\":\"x\"}"}"#,
            r#"{"type":"response.done","response":{"id":"resp-a","status":"completed"}}"#,
        ] {
            peer.incoming
                .unbounded_send(Ok(RealtimeFrame::text(message)))
                .unwrap();
        }
        let session_task = async move {
            let _created = events.next().await;
            let _conversation = events.next().await;
            let _updated = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::SpeechStarted {
                    item_id: Some(item_id),
                    audio_start_ms: Some(120),
                    ..
                }) if item_id == "user-a"
            ));
            assert_eq!(
                events.next().await,
                Some(XaiRealtimeEvent::InputTranscriptionUpdated {
                    item_id: Some("user-a".into()),
                    transcript: "hel".into(),
                    native: json!({
                        "type": "conversation.item.input_audio_transcription.updated",
                        "item_id": "user-a",
                        "transcript": "hel"
                    }),
                })
            );
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::InputTranscriptionCompleted { transcript, .. }) if transcript == "hello"
            ));
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::TextDelta { delta, .. }) if delta == "hi"
            ));
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::FunctionCallArgumentsDone {
                    call_id: Some(call_id),
                    arguments: Some(arguments),
                    ..
                }) if call_id == "call-a" && arguments == r#"{"q":"x"}"#
            ));
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::ResponseDone { response_id: Some(id), .. }) if id == "resp-a"
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        assert_eq!(
            sent_json(&peer.sent)[0]["session"]["audio"]["input"]["transcription"]["model"],
            "grok-transcribe"
        );
    });
}

#[test]
fn parallel_tool_outputs_wait_for_one_explicit_and_deferred_continuation() {
    block_on(async {
        let (transport, mut peer) = fake_transport(&CREATED_PREFACE);
        let (session, driver) = XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let sent = peer.sent.clone();
        let sent_after = sent.clone();
        for message in [
            r#"{"type":"response.function_call_arguments.done","response_id":"resp-1","call_id":"call-1","name":"lookup","arguments":"{\"q\":\"first\"}"}"#,
            r#"{"type":"response.function_call_arguments.done","response_id":"resp-1","call_id":"call-2","name":"lookup","arguments":"{\"q\":\"second\"}"}"#,
            r#"{"type":"response.done","response":{"id":"resp-1","status":"completed"}}"#,
        ] {
            peer.incoming
                .unbounded_send(Ok(RealtimeFrame::text(message)))
                .unwrap();
        }
        let (results_sent_tx, results_sent_rx) = oneshot::channel();
        let (playback_done_tx, playback_done_rx) = oneshot::channel();

        let session_task = async move {
            let _session_created = events.next().await;
            let _conversation_created = events.next().await;
            let _session_updated = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::FunctionCallArgumentsDone {
                    call_id: Some(call_id), ..
                }) if call_id == "call-1"
            ));
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::FunctionCallArgumentsDone {
                    call_id: Some(call_id), ..
                }) if call_id == "call-2"
            ));
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::ResponseDone {
                    response_id: Some(response_id), ..
                }) if response_id == "resp-1"
            ));

            let setup = peer.sent_notifications.next().await.unwrap();
            assert_eq!(sent_frame_json(setup)["type"], "session.update");
            control
                .send(RealtimeInput::ToolResults {
                    results: vec![
                        RealtimeToolResult {
                            call_id: "call-1".into(),
                            output: json!({"found": "first"}),
                        },
                        RealtimeToolResult {
                            call_id: "call-2".into(),
                            output: json!({"found": "second"}),
                        },
                    ],
                })
                .unwrap();
            let first = peer.sent_notifications.next().await.unwrap();
            let second = peer.sent_notifications.next().await.unwrap();
            assert_eq!(
                sent_frame_json(first),
                json!({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "{\"found\":\"first\"}"
                    }
                })
            );
            assert_eq!(
                sent_frame_json(second),
                json!({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call-2",
                        "output": "{\"found\":\"second\"}"
                    }
                })
            );
            assert_eq!(sent.lock().unwrap().len(), 3, "no response.create yet");
            results_sent_tx.send(()).unwrap();

            // The host waits for local audio playback before releasing this
            // gate and asking xAI to continue.
            playback_done_rx.await.unwrap();
            control.send(RealtimeInput::ContinueResponse).unwrap();
            let continuation = peer.sent_notifications.next().await.unwrap();
            assert_eq!(
                sent_frame_json(continuation),
                json!({ "type": "response.create" })
            );
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let host_playback = async move {
            results_sent_rx.await.unwrap();
            playback_done_tx.send(()).unwrap();
        };
        let ((), (), driver_result) = futures::join!(session_task, host_playback, driver.run());
        driver_result.unwrap();
        assert_eq!(
            sent_json(&sent_after),
            vec![
                json!({"type":"session.update", "session": {
                    "voice":"eve",
                    "turn_detection":{"type":"server_vad"},
                    "audio": {
                        "input":{"format":{"type":"audio/pcm","rate":24000},"transport":"json"},
                        "output":{"format":{"type":"audio/pcm","rate":24000},"transport":"json"}
                    }
                }}),
                json!({
                    "type":"conversation.item.create",
                    "item":{"type":"function_call_output","call_id":"call-1","output":"{\"found\":\"first\"}"}
                }),
                json!({
                    "type":"conversation.item.create",
                    "item":{"type":"function_call_output","call_id":"call-2","output":"{\"found\":\"second\"}"}
                }),
                json!({"type":"response.create"}),
            ]
        );
    });
}

#[test]
fn single_tool_result_is_submitted_without_implicit_continuation() {
    block_on(async {
        let (transport, mut peer) = fake_transport(&CREATED_PREFACE);
        let (session, driver) = XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let sent = peer.sent.clone();
        let sent_after = sent.clone();
        let session_task = async move {
            let _session_created = events.next().await;
            let _conversation_created = events.next().await;
            let _session_updated = events.next().await;
            assert_eq!(
                sent_frame_json(peer.sent_notifications.next().await.unwrap())["type"],
                "session.update"
            );
            control
                .send(RealtimeInput::ToolResult {
                    call_id: "call-single".into(),
                    output: json!({"ok": true}),
                })
                .unwrap();
            assert_eq!(
                sent_frame_json(peer.sent_notifications.next().await.unwrap()),
                json!({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call-single",
                        "output": "{\"ok\":true}"
                    }
                })
            );
            assert_eq!(sent.lock().unwrap().len(), 2);
            control.close(RealtimeClose::normal("done")).await.unwrap();
            assert!(matches!(
                events.next().await,
                Some(XaiRealtimeEvent::Realtime(RealtimeEvent::Closed { .. }))
            ));
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        assert_eq!(sent_json(&sent_after).len(), 2);
    });
}

#[test]
fn invalid_audio_and_tool_result_inputs_are_rejected_before_queueing() {
    block_on(async {
        let (transport, peer) = fake_transport(&CREATED_PREFACE);
        let (session, _driver) = XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, _events) = session.into_parts();
        assert!(matches!(
            control.send(RealtimeInput::Audio {
                data: Bytes::from_static(&[1, 2]),
                format: RealtimeAudioFormat::Pcm16 {
                    sample_rate_hz: 16_000,
                },
            }),
            Err(RealtimeError::InvalidInput { .. })
        ));
        assert!(matches!(
            control.send(RealtimeInput::ToolResult {
                call_id: "".into(),
                output: json!({ "ok": true }),
            }),
            Err(RealtimeError::InvalidInput { .. })
        ));
        assert!(matches!(
            control.send(RealtimeInput::ToolResults { results: vec![] }),
            Err(RealtimeError::InvalidInput { .. })
        ));
        assert!(matches!(
            control.send(RealtimeInput::ToolResults {
                results: vec![
                    RealtimeToolResult {
                        call_id: "duplicate".into(),
                        output: json!({"one": true}),
                    },
                    RealtimeToolResult {
                        call_id: "duplicate".into(),
                        output: json!({"two": true}),
                    },
                ],
            }),
            Err(RealtimeError::InvalidInput { .. })
        ));
        assert!(matches!(
            control.send(RealtimeInput::ToolResults {
                results: (0..=MAX_REALTIME_TOOL_RESULTS)
                    .map(|index| RealtimeToolResult {
                        call_id: format!("call-{index}"),
                        output: Value::Null,
                    })
                    .collect(),
            }),
            Err(RealtimeError::InvalidInput { .. })
        ));
        assert!(matches!(
            control.send(RealtimeInput::ToolResults {
                results: vec![RealtimeToolResult {
                    call_id: "large-output".into(),
                    output: Value::String("x".repeat(20 * 1024)),
                }],
            }),
            Err(RealtimeError::FrameTooLarge { .. })
        ));
        assert_eq!(
            sent_json(&peer.sent).len(),
            1,
            "only session.update was sent"
        );
    });
}

#[test]
fn setup_error_or_missing_session_update_ack_does_not_return_a_session() {
    block_on(async {
        let (transport, peer) = fake_transport(&[
            r#"{"type":"session.created","session":{"id":"sess-1"}}"#,
            r#"{"type":"error","error":{"code":"invalid_request","message":"bad setup"}}"#,
        ]);
        let error = XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .err()
        .expect("provider setup errors must fail connect");
        assert!(matches!(error, RealtimeError::Transport { .. }));
        assert_eq!(sent_json(&peer.sent).len(), 1);

        let (transport, peer) = fake_transport(&[
            r#"{"type":"session.created","session":{"id":"sess-2"}}"#,
            r#"{"type":"conversation.created","conversation":{"id":"conv-2"}}"#,
        ]);
        let peer_count = peer.connect_count.clone();
        drop(peer);
        let error = XaiRealtimeSession::connect(
            as_transport(transport),
            request(),
            XaiRealtimeConfig::default(),
            limits(),
        )
        .await
        .err()
        .expect("session.updated is required before connect returns");
        assert_eq!(error, RealtimeError::UnexpectedRemoteClose);
        assert_eq!(*peer_count.lock().unwrap(), 1);
    });
}
