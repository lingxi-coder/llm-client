use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::mpsc,
    executor::block_on,
    stream::{self, BoxStream},
    StreamExt,
};
use lingxi_llm_client::providers::google::live::GeminiLiveConfig;
use lingxi_llm_client::providers::google::live::GeminiLiveEvent;
use lingxi_llm_client::providers::google::live::GeminiLiveSession;
use lingxi_llm_client::realtime::{
    RealtimeAudioFormat, RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError,
    RealtimeEvent, RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSink, RealtimeTransport,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    outbound: mpsc::UnboundedSender<RealtimeFrame>,
    connect_count: Arc<AtomicUsize>,
}

struct FakePeer {
    incoming: mpsc::UnboundedSender<Incoming>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    outbound: mpsc::UnboundedReceiver<RealtimeFrame>,
    connect_count: Arc<AtomicUsize>,
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    outbound: mpsc::UnboundedSender<RealtimeFrame>,
}

fn fake_transport(preface: &[&str]) -> (Arc<FakeTransport>, FakePeer) {
    let (incoming_tx, incoming_rx) = mpsc::unbounded();
    let prefix = preface
        .iter()
        .map(|message| Ok(RealtimeFrame::text((*message).to_owned())))
        .collect::<Vec<_>>();
    let incoming: BoxStream<'static, Incoming> = stream::iter(prefix).chain(incoming_rx).boxed();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let connect_count = Arc::new(AtomicUsize::new(0));
    let (outbound_tx, outbound_rx) = mpsc::unbounded();
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(incoming)),
            sent: sent.clone(),
            outbound: outbound_tx.clone(),
            connect_count: connect_count.clone(),
        }),
        FakePeer {
            incoming: incoming_tx,
            sent,
            outbound: outbound_rx,
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
        assert!(request.endpoint.starts_with("wss://"));
        assert_eq!(request.max_frame_bytes, 16 * 1024);
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        let incoming = self
            .incoming
            .lock()
            .unwrap()
            .take()
            .expect("one session per fake transport");
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
                outbound: self.outbound.clone(),
            }),
            inbound: incoming,
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.sent.lock().unwrap().push(frame.clone());
        let _ = self.outbound.unbounded_send(frame);
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}

fn request() -> RealtimeConnectRequest {
    RealtimeConnectRequest {
        endpoint: "wss://generativelanguage.googleapis.com/ws/live?key=private-key".into(),
        headers: vec![],
        max_frame_bytes: 1,
    }
}

fn config() -> GeminiLiveConfig {
    GeminiLiveConfig::new("gemini-3.8-live", "google-account-1")
}

fn limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 16,
        max_frame_bytes: 16 * 1024,
    }
}

fn transport_trait(transport: Arc<FakeTransport>) -> Arc<dyn RealtimeTransport> {
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

async fn next_outbound_json(peer: &mut FakePeer) -> Value {
    let frame = peer.outbound.next().await.expect("outbound frame");
    let RealtimeFrame::Text(bytes) = frame else {
        panic!("expected a JSON text frame");
    };
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn setup_is_the_first_message_and_session_is_not_returned_until_ready() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();

        let session_task = async move {
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::SessionReady))
            );
            control.send(RealtimeInput::Text("hello".into())).unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::Closed {
                    code: 1000,
                    reason: "done".into(),
                }))
            );
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(sent[0]["setup"]["model"], "models/gemini-3.8-live");
        assert_eq!(
            sent[0]["setup"]["generationConfig"]["responseModalities"][0],
            "AUDIO"
        );
        assert_eq!(sent[1], json!({ "realtimeInput": { "text": "hello" } }));
        assert_eq!(peer.connect_count.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn pcm_audio_and_audio_stream_end_use_gemini_live_wire_shapes() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        let audio = Bytes::from_static(&[1, 2, 3, 4]);

        let session_task = async move {
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::SessionReady))
            );
            control
                .send(RealtimeInput::Audio {
                    data: audio,
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000,
                    },
                })
                .unwrap();
            control.send(RealtimeInput::CommitAudio).unwrap();
            assert!(matches!(
                control.interrupt(),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(
            sent[1]["realtimeInput"]["audio"]["mimeType"],
            "audio/pcm;rate=16000"
        );
        assert_eq!(sent[1]["realtimeInput"]["audio"]["data"], "AQIDBA==");
        assert_eq!(
            sent[2],
            json!({ "realtimeInput": { "audioStreamEnd": true } })
        );
    });
}

#[test]
fn manual_activity_config_brackets_text_and_maps_interrupt_and_commit() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let mut setup = config();
        setup.realtime_input_config = Some(json!({
            "automaticActivityDetection": { "disabled": true }
        }));
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), setup, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();

        let session_task = async move {
            let _ready = events.next().await;
            control.send(RealtimeInput::Text("spoken".into())).unwrap();
            control.interrupt().unwrap();
            control.send(RealtimeInput::CommitAudio).unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(
            sent[0]["setup"]["realtimeInputConfig"]["automaticActivityDetection"]["disabled"],
            true
        );
        assert_eq!(sent[1], json!({ "realtimeInput": { "activityStart": {} } }));
        assert_eq!(sent[2], json!({ "realtimeInput": { "text": "spoken" } }));
        assert_eq!(sent[3], json!({ "realtimeInput": { "activityEnd": {} } }));
        assert_eq!(sent[4], json!({ "realtimeInput": { "activityStart": {} } }));
        assert_eq!(sent[5], json!({ "realtimeInput": { "activityEnd": {} } }));
    });
}

#[test]
fn server_content_normalizes_audio_text_interruption_and_completion() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"serverContent":{"modelTurn":{"parts":[{"text":"hello"},{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQID"}}]},"interrupted":true,"turnComplete":true}}"#,
            )))
            .unwrap();

        let session_task = async move {
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::SessionReady))
            );
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::TextDelta {
                    text: "hello".into(),
                    item_id: None,
                    final_chunk: true,
                }))
            );
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::AudioDelta {
                    data: Bytes::from_static(&[1, 2, 3]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 24_000,
                    },
                    item_id: None,
                }))
            );
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::Interrupted))
            );
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::TurnCompleted {
                    turn_id: None,
                    status: Some("interrupted".into()),
                }))
            );
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
    });
}

#[test]
fn function_call_results_match_the_provider_call_id_and_name() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"toolCall":{"functionCalls":[{"id":"call-1","name":"lookup","args":{"q":"x"}}]}}"#,
            )))
            .unwrap();

        let session_task = async move {
            let _ready = events.next().await;
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::ToolCall {
                    call_id: "call-1".into(),
                    name: "lookup".into(),
                    arguments: json!({ "q": "x" }),
                }))
            );
            control
                .send(RealtimeInput::ToolResult {
                    call_id: "call-1".into(),
                    output: json!({ "ok": true }),
                })
                .unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(
            sent[1],
            json!({
                "toolResponse": {
                    "functionResponses": [{
                        "id": "call-1",
                        "name": "lookup",
                        "response": { "ok": true }
                    }]
                }
            })
        );
    });
}

#[test]
fn images_encode_as_video_frames_and_invalid_frames_are_rejected_locally() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();

        let session_task = async move {
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::SessionReady))
            );
            control
                .send(RealtimeInput::Image {
                    data: Bytes::from_static(&[1, 2, 3]),
                    mime_type: "image/jpeg".into(),
                })
                .unwrap();
            assert!(matches!(
                control.send(RealtimeInput::Image {
                    data: Bytes::new(),
                    mime_type: "image/jpeg".into(),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            assert!(matches!(
                control.send(RealtimeInput::Image {
                    data: Bytes::from_static(&[1]),
                    mime_type: "image/gif".into(),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(
            sent[1],
            json!({
                "realtimeInput": {
                    "video": { "mimeType": "image/jpeg", "data": "AQID" }
                }
            })
        );
        assert_eq!(sent.len(), 2, "rejected images must not reach the sink");
    });
}

#[test]
fn malformed_function_call_batch_does_not_register_a_partial_result_mapping() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"toolCall":{"functionCalls":[{"id":"call-1","name":"lookup","args":{}},{"id":"call-2","args":{}}]}}"#,
            )))
            .unwrap();
        let incoming = peer.incoming.clone();

        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();

        let session_task = async move {
            let _ready = events.next().await;
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::ProviderEvent {
                    name: "toolCall".into(),
                    native: json!({
                        "toolCall": { "functionCalls": [
                            { "id": "call-1", "name": "lookup", "args": {} },
                            { "id": "call-2", "args": {} }
                        ] }
                    }),
                }))
            );
            assert!(matches!(
                control.send(RealtimeInput::ToolResult {
                    call_id: "call-1".into(),
                    output: json!({ "ok": true }),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            incoming
                .unbounded_send(Ok(RealtimeFrame::text(
                    r#"{"toolCall":{"functionCalls":[{"id":"call-1","name":"lookup","args":{}}]}}"#,
                )))
                .unwrap();
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::ToolCall {
                    call_id: "call-1".into(),
                    name: "lookup".into(),
                    arguments: json!({}),
                }))
            );
            control
                .send(RealtimeInput::ToolResult {
                    call_id: "call-1".into(),
                    output: json!({ "ok": true }),
                })
                .unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        assert_eq!(
            sent_json(&peer.sent)[1]["toolResponse"]["functionResponses"][0]["id"],
            "call-1"
        );
    });
}

#[test]
fn invalid_batched_tool_output_is_atomic_and_successful_batch_cleans_all_names() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"toolCall":{"functionCalls":[{"id":"call-1","name":"lookup","args":{}},{"id":"call-2","name":"search","args":{}}]}}"#,
            )))
            .unwrap();
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();

        let session_task = async move {
            let _ready = events.next().await;
            for (call_id, name) in [("call-1", "lookup"), ("call-2", "search")] {
                assert_eq!(
                    events.next().await,
                    Some(GeminiLiveEvent::Realtime(RealtimeEvent::ToolCall {
                        call_id: call_id.into(),
                        name: name.into(),
                        arguments: json!({}),
                    }))
                );
            }
            assert!(matches!(
                control.send(RealtimeInput::ToolResults {
                    results: vec![
                        lingxi_llm_client::realtime::RealtimeToolResult {
                            call_id: "call-1".into(),
                            output: json!({ "ok": true }),
                        },
                        lingxi_llm_client::realtime::RealtimeToolResult {
                            call_id: "call-2".into(),
                            output: Value::Null,
                        },
                    ],
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control
                .send(RealtimeInput::ToolResults {
                    results: vec![
                        lingxi_llm_client::realtime::RealtimeToolResult {
                            call_id: "call-1".into(),
                            output: json!({ "ok": true }),
                        },
                        lingxi_llm_client::realtime::RealtimeToolResult {
                            call_id: "call-2".into(),
                            output: json!({ "found": 3 }),
                        },
                    ],
                })
                .unwrap();
            assert!(matches!(
                control.send(RealtimeInput::ToolResult {
                    call_id: "call-1".into(),
                    output: json!({}),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        assert_eq!(
            sent_json(&peer.sent)[1],
            json!({
                "toolResponse": {
                    "functionResponses": [
                        { "id": "call-1", "name": "lookup", "response": { "ok": true } },
                        { "id": "call-2", "name": "search", "response": { "found": 3 } }
                    ]
                }
            })
        );
    });
}

#[test]
fn accepted_results_release_call_names_and_queue_full_keeps_them_retryable() {
    block_on(async {
        let (transport, mut peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"toolCall":{"functionCalls":[{"id":"call-1","name":"lookup","args":{}}]}}"#,
            )))
            .unwrap();
        let mut small_queue = limits();
        small_queue.outbound_capacity = 1;
        let (session, driver) = GeminiLiveSession::connect(
            transport_trait(transport),
            request(),
            config(),
            small_queue,
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        assert_eq!(
            next_outbound_json(&mut peer).await["setup"]["model"],
            "models/gemini-3.8-live"
        );
        let sent = peer.sent.clone();

        let session_task = async move {
            let _ready = events.next().await;
            let _call = events.next().await;
            control
                .send(RealtimeInput::Text("occupy the only queue slot".into()))
                .unwrap();
            assert_eq!(
                control.send(RealtimeInput::ToolResult {
                    call_id: "call-1".into(),
                    output: json!({ "ok": true }),
                }),
                Err(RealtimeError::QueueFull)
            );
            let text_frame = next_outbound_json(&mut peer).await;
            assert_eq!(
                text_frame["realtimeInput"]["text"],
                "occupy the only queue slot"
            );
            control
                .send(RealtimeInput::ToolResult {
                    call_id: "call-1".into(),
                    output: json!({ "ok": true }),
                })
                .unwrap();
            assert_eq!(
                next_outbound_json(&mut peer).await["toolResponse"]["functionResponses"][0]["name"],
                "lookup"
            );
            assert!(matches!(
                control.send(RealtimeInput::ToolResult {
                    call_id: "call-1".into(),
                    output: json!({ "again": true }),
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        assert_eq!(sent.lock().unwrap().len(), 3);
    });
}

#[test]
fn more_than_one_thousand_completed_calls_do_not_exhaust_name_tracking() {
    block_on(async {
        let (transport, mut peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        let _setup = next_outbound_json(&mut peer).await;

        let session_task = async move {
            assert_eq!(
                events.next().await,
                Some(GeminiLiveEvent::Realtime(RealtimeEvent::SessionReady))
            );
            for index in 0..1_025 {
                let call_id = format!("call-{index}");
                peer.incoming
                    .unbounded_send(Ok(RealtimeFrame::text(
                        json!({ "toolCall": { "functionCalls": [{
                            "id": call_id,
                            "name": "lookup",
                            "args": {}
                        }] } })
                        .to_string(),
                    )))
                    .unwrap();
                let Some(GeminiLiveEvent::Realtime(RealtimeEvent::ToolCall {
                    call_id: received_id,
                    ..
                })) = events.next().await
                else {
                    panic!("expected tool call {index}");
                };
                assert_eq!(received_id, call_id);
                control
                    .send(RealtimeInput::ToolResult {
                        call_id: call_id.clone(),
                        output: json!({ "ok": true }),
                    })
                    .unwrap();
                let response = next_outbound_json(&mut peer).await;
                assert_eq!(
                    response["toolResponse"]["functionResponses"][0]["id"],
                    call_id
                );
            }
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
    });
}

#[test]
fn resumption_handles_are_typed_scoped_and_redacted() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let mut setup = config();
        setup.enable_session_resumption = true;
        let (session, driver) =
            GeminiLiveSession::connect(transport_trait(transport), request(), setup, limits())
                .await
                .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"sessionResumptionUpdate":{"newHandle":"resume-secret","resumable":true}}"#,
            )))
            .unwrap();

        let session_task = async move {
            let _ready = events.next().await;
            let update = events.next().await;
            let Some(GeminiLiveEvent::SessionResumptionUpdate(update)) = update else {
                panic!("expected a typed resumption update");
            };
            let handle = update.handle.clone().expect("resumable handle");
            control.close(RealtimeClose::normal("done")).await.unwrap();
            let _closed = events.next().await;
            (update, handle)
        };
        let ((update, handle), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        assert!(update.resumable);
        assert_eq!(handle.model(), "models/gemini-3.8-live");
        assert_eq!(handle.credential_scope(), "google-account-1");
        assert!(!format!("{handle:?}").contains("resume-secret"));
        assert!(serde_json::to_string(&handle)
            .unwrap()
            .contains("resume-secret"));

        let (next_transport, _next_peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let mut wrong_scope = config();
        wrong_scope.credential_scope = "google-account-2".into();
        wrong_scope.enable_session_resumption = true;
        wrong_scope.resume_handle = Some(handle.clone());
        let error = GeminiLiveSession::connect(
            transport_trait(next_transport.clone()),
            request(),
            wrong_scope,
            limits(),
        )
        .await
        .err()
        .expect("cross-account resume must fail before network");
        assert!(matches!(error, RealtimeError::InvalidConfig { .. }));
        assert_eq!(next_transport.connect_count.load(Ordering::SeqCst), 0);

        let (resume_transport, resume_peer) = fake_transport(&[r#"{"setupComplete":{}}"#]);
        let mut resume_config = config();
        resume_config.enable_session_resumption = true;
        resume_config.resume_handle = Some(handle);
        let (resumed, _driver) = GeminiLiveSession::connect(
            transport_trait(resume_transport),
            request(),
            resume_config,
            limits(),
        )
        .await
        .unwrap();
        let sent = sent_json(&resume_peer.sent);
        assert_eq!(
            sent[0]["setup"]["sessionResumption"]["handle"],
            "resume-secret"
        );
        drop(resumed);
    });
}

#[test]
fn setup_failure_is_reported_without_returning_a_live_session() {
    block_on(async {
        let (transport, peer) = fake_transport(&[r#"{"serverContent":{"waitingForInput":true}}"#]);
        let (connect_count, sent) = (peer.connect_count.clone(), peer.sent.clone());
        drop(peer);
        let error =
            GeminiLiveSession::connect(transport_trait(transport), request(), config(), limits())
                .await
                .err()
                .expect("setupComplete is required before returning a session");

        assert_eq!(connect_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            error,
            RealtimeError::UnexpectedRemoteClose,
            "the server stream ended before setupComplete"
        );
        assert_eq!(
            sent_json(&sent)[0]["setup"]["model"],
            "models/gemini-3.8-live"
        );
    });
}
