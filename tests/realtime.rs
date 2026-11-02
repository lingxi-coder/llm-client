use async_trait::async_trait;
use bytes::Bytes;
use futures::{channel::mpsc, executor::block_on, stream::BoxStream, StreamExt};
use lingxi_llm_client::providers::openai::realtime::OpenAiAudioFormat;
use lingxi_llm_client::providers::openai::realtime::OpenAiOutputMode;
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeCodec;
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeConfig;
use lingxi_llm_client::realtime;
use lingxi_llm_client::realtime::{
    RealtimeClose, RealtimeCodec, RealtimeConnectRequest, RealtimeConnection, RealtimeControl,
    RealtimeDriver, RealtimeError, RealtimeEvent, RealtimeFrame, RealtimeLimits,
    RealtimeOrderedEvent, RealtimeSession, RealtimeSink, RealtimeTransport,
};
use std::sync::{Arc, Mutex};

type IncomingFrame = Result<RealtimeFrame, RealtimeError>;
type IncomingReceiver = mpsc::UnboundedReceiver<IncomingFrame>;

#[derive(Clone)]
struct FakeTransport {
    incoming: Arc<Mutex<Option<IncomingReceiver>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    closed: Arc<Mutex<Option<RealtimeClose>>>,
}

struct FakePeer {
    incoming: mpsc::UnboundedSender<IncomingFrame>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    closed: Arc<Mutex<Option<RealtimeClose>>>,
}

impl FakeTransport {
    fn new() -> (Self, FakePeer) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded();
        let sent = Arc::new(Mutex::new(Vec::new()));
        let closed = Arc::new(Mutex::new(None));
        (
            Self {
                incoming: Arc::new(Mutex::new(Some(incoming_rx))),
                sent: sent.clone(),
                closed: closed.clone(),
            },
            FakePeer {
                incoming: incoming_tx,
                sent,
                closed,
            },
        )
    }
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    closed: Arc<Mutex<Option<RealtimeClose>>>,
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

    async fn close(&mut self, close: RealtimeClose) -> Result<(), RealtimeError> {
        *self.closed.lock().unwrap() = Some(close);
        Ok(())
    }
}

#[test]
fn injected_sinks_without_ping_support_return_an_explicit_non_sending_error() {
    let sent = Arc::new(Mutex::new(Vec::new()));
    let mut sink = FakeSink {
        sent: sent.clone(),
        closed: Arc::new(Mutex::new(None)),
    };

    let error = block_on(sink.ping(Bytes::from_static(b"host-ping")))
        .expect_err("the test sink must report unsupported Ping capability");

    assert!(matches!(error, RealtimeError::InvalidInput { .. }));
    assert!(error
        .to_string()
        .contains("does not support explicit WebSocket Ping"));
    assert!(sent.lock().unwrap().is_empty());
}

#[async_trait]
impl RealtimeTransport for FakeTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        assert!(request.endpoint.starts_with("wss://"));
        assert_eq!(request.max_frame_bytes, 4096);
        let incoming = self
            .incoming
            .lock()
            .unwrap()
            .take()
            .expect("one connection per fake transport");
        let incoming: BoxStream<'static, Result<RealtimeFrame, RealtimeError>> = incoming.boxed();
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
                closed: self.closed.clone(),
            }),
            inbound: incoming,
        })
    }
}

fn request() -> RealtimeConnectRequest {
    RealtimeConnectRequest {
        endpoint: "wss://example.invalid/realtime?credential=secret".into(),
        headers: vec![("Authorization".into(), "Bearer secret".into())],
        max_frame_bytes: 4096,
    }
}

async fn connected(
    limits: RealtimeLimits,
) -> (
    RealtimeControl,
    realtime::RealtimeEvents,
    RealtimeDriver,
    FakePeer,
) {
    let (transport, peer) = FakeTransport::new();
    let (session, driver) = RealtimeSession::connect(
        &transport,
        request(),
        Arc::new(OpenAiRealtimeCodec::default()),
        limits,
    )
    .await
    .unwrap();
    let (control, events) = session.into_parts();
    (control, events, driver, peer)
}

fn limits(capacity: usize, max_frame_bytes: usize) -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: capacity,
        event_capacity: 8,
        max_frame_bytes,
    }
}

#[tokio::test]
async fn input_fence_orders_raw_frames_queued_before_the_driver_starts() {
    let (control, events, driver, peer) = connected(limits(4, 4096)).await;
    peer.incoming.unbounded_send(Ok(RealtimeFrame::text(
        r#"{"type":"conversation.item.input_audio_transcription.delta","item_id":"old","delta":"unowned"}"#,
    ))).unwrap();
    control
        .send_with_fence(realtime::RealtimeInput::Text("new input".into()), 17)
        .unwrap();
    let task = tokio::spawn(driver.run());
    let mut events = events.into_ordered();

    assert!(
        matches!(events.next().await, Some(RealtimeOrderedEvent::Provider(
        RealtimeEvent::Transcript { item_id: Some(id), .. }
    )) if id == "old")
    );
    assert!(
        matches!(events.next().await, Some(RealtimeOrderedEvent::Provider(
        RealtimeEvent::ProviderEvent { name, .. }
    )) if name == "conversation.item.input_audio_transcription.delta")
    );
    assert_eq!(
        events.next().await,
        Some(RealtimeOrderedEvent::InputSent { id: 17 })
    );
    assert!(
        peer.sent.lock().unwrap().iter().any(|frame| match frame {
            RealtimeFrame::Text(bytes) => String::from_utf8_lossy(bytes).contains("new input"),
            _ => false,
        }),
        "the driver must send the input before emitting the marker"
    );

    control
        .abort(RealtimeClose::normal("test finished"))
        .await
        .unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn input_fence_drains_under_backpressure_and_fails_closed_on_backlog_limit() {
    let mut bounded = limits(4, 4096);
    bounded.event_capacity = 2;
    let (control, events, driver, peer) = connected(bounded).await;
    let initial_sends = peer.sent.lock().unwrap().len();
    for _ in 0..=realtime::MAX_REALTIME_INPUT_FENCE_FRAMES {
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(r#"{"type":"session.updated"}"#)))
            .unwrap();
    }
    control
        .send_with_fence(realtime::RealtimeInput::Text("must not send".into()), 18)
        .unwrap();
    let task = tokio::spawn(driver.run());
    let mut events = events.into_ordered();
    let mut ready_count = 0;
    let mut saw_limit = false;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), events.next())
        .await
        .unwrap()
    {
        match event {
            RealtimeOrderedEvent::Provider(RealtimeEvent::SessionReady) => ready_count += 1,
            RealtimeOrderedEvent::Provider(RealtimeEvent::ProviderError {
                code: Some(code),
                ..
            }) if code == "input_fence_backlog_full" => saw_limit = true,
            RealtimeOrderedEvent::InputSent { .. } => {
                panic!("overflow must not acknowledge the input")
            }
            _ => {}
        }
    }
    let error = task
        .await
        .unwrap()
        .expect_err("the fence must reject an unbounded ready stream");
    assert!(error.to_string().contains("input fence inbound backlog"));
    assert_eq!(
        ready_count,
        realtime::MAX_REALTIME_INPUT_FENCE_FRAMES,
        "previous frames must drain while the receiver keeps reading"
    );
    assert!(saw_limit);
    assert_eq!(
        peer.sent.lock().unwrap().len(),
        initial_sends,
        "the input must not reach the transport"
    );
}

#[tokio::test]
async fn input_fence_accepts_backlog_larger_than_event_queue() {
    let mut bounded = limits(4, 4096);
    bounded.event_capacity = 2;
    let (control, events, driver, peer) = connected(bounded).await;
    for _ in 0..3 {
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(r#"{"type":"session.updated"}"#)))
            .unwrap();
    }
    control
        .send_with_fence(realtime::RealtimeInput::Text("input".into()), 19)
        .unwrap();
    let task = tokio::spawn(driver.run());
    let mut events = events.into_ordered();
    let mut ready_count = 0;
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(5), events.next())
            .await
            .unwrap()
        {
            Some(RealtimeOrderedEvent::Provider(RealtimeEvent::SessionReady)) => ready_count += 1,
            Some(RealtimeOrderedEvent::InputSent { id: 19 }) => break,
            Some(_) => {}
            None => panic!("driver closed before its input marker"),
        }
    }
    assert_eq!(ready_count, 3);
    control
        .abort(RealtimeClose::normal("test finished"))
        .await
        .unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn input_fence_rejects_invalid_and_full_queue_without_extra_markers() {
    let (control, events, driver, _peer) = connected(limits(1, 4096)).await;
    assert!(matches!(
        control.send_with_fence(realtime::RealtimeInput::Interrupt, 1),
        Err(RealtimeError::InvalidInput { .. })
    ));
    assert!(matches!(
        control.send_with_fence(realtime::RealtimeInput::Text("x".repeat(5000)), 2),
        Err(RealtimeError::FrameTooLarge { .. })
    ));
    control
        .send_with_fence(realtime::RealtimeInput::Text("accepted".into()), 3)
        .unwrap();
    assert_eq!(
        control.send_with_fence(realtime::RealtimeInput::Text("full".into()), 4),
        Err(RealtimeError::QueueFull)
    );
    let task = tokio::spawn(driver.run());
    let mut events = events.into_ordered();
    assert_eq!(
        events.next().await,
        Some(RealtimeOrderedEvent::InputSent { id: 3 })
    );
    control
        .abort(RealtimeClose::normal("test finished"))
        .await
        .unwrap();
    while let Some(event) = events.next().await {
        assert!(!matches!(event, RealtimeOrderedEvent::InputSent { .. }));
    }
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn plain_receiver_rejects_fenced_input_instead_of_silently_losing_marker() {
    let (control, mut events, driver, _peer) = connected(limits(2, 4096)).await;
    control
        .send_with_fence(realtime::RealtimeInput::Text("input".into()), 5)
        .unwrap();
    let task = tokio::spawn(driver.run());
    assert!(
        matches!(events.next().await, Some(RealtimeEvent::ConnectionInterrupted { message })
        if message.contains("ordered realtime event receiver"))
    );
    assert_eq!(events.next().await, None);
    control
        .abort(RealtimeClose::normal("test finished"))
        .await
        .unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn input_fence_send_failure_reports_interruption_without_marker() {
    struct FailingSink;
    #[async_trait]
    impl RealtimeSink for FailingSink {
        async fn send(&mut self, _: RealtimeFrame) -> Result<(), RealtimeError> {
            Err(RealtimeError::Transport {
                message: "fixture send failed".into(),
            })
        }
        async fn ping(&mut self, _: Bytes) -> Result<(), RealtimeError> {
            Ok(())
        }
        fn abort(&mut self) {}
        async fn close(&mut self, _: RealtimeClose) -> Result<(), RealtimeError> {
            Ok(())
        }
    }
    struct FailingTransport(Mutex<Option<IncomingReceiver>>);
    #[async_trait]
    impl RealtimeTransport for FailingTransport {
        async fn connect(
            &self,
            _: RealtimeConnectRequest,
        ) -> Result<RealtimeConnection, RealtimeError> {
            Ok(RealtimeConnection {
                outbound: Box::new(FailingSink),
                inbound: self.0.lock().unwrap().take().unwrap().boxed(),
            })
        }
    }
    struct SimpleCodec;
    impl RealtimeCodec for SimpleCodec {
        fn encode(&self, _: &realtime::RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
            Ok(vec![RealtimeFrame::text("input")])
        }
        fn decode(&self, _: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
            Ok(Vec::new())
        }
    }
    let (_peer, incoming) = mpsc::unbounded();
    let (session, driver) = RealtimeSession::connect(
        &FailingTransport(Mutex::new(Some(incoming))),
        request(),
        Arc::new(SimpleCodec),
        limits(2, 4096),
    )
    .await
    .unwrap();
    let (control, events) = session.into_parts();
    control
        .send_with_fence(realtime::RealtimeInput::Text("input".into()), 6)
        .unwrap();
    let task = tokio::spawn(driver.run());
    let mut events = events.into_ordered();
    assert!(
        matches!(events.next().await, Some(RealtimeOrderedEvent::Provider(
        RealtimeEvent::ConnectionInterrupted { message }
    )) if message.contains("fixture send failed"))
    );
    assert_eq!(
        events.next().await,
        None,
        "failed transport sends must never produce an input marker"
    );
    assert!(matches!(
        task.await.unwrap(),
        Err(RealtimeError::Transport { .. })
    ));
}

#[tokio::test]
async fn abort_preempts_a_fence_blocked_by_event_backpressure() {
    struct CountingCodec(Arc<std::sync::atomic::AtomicUsize>);
    impl RealtimeCodec for CountingCodec {
        fn encode(&self, _: &realtime::RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
            Ok(vec![RealtimeFrame::text("input")])
        }
        fn decode(&self, _: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![RealtimeEvent::SessionReady])
        }
    }
    let decoded = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (transport, peer) = FakeTransport::new();
    let mut bounded = limits(2, 4096);
    bounded.event_capacity = 2;
    let (session, driver) = RealtimeSession::connect(
        &transport,
        request(),
        Arc::new(CountingCodec(decoded.clone())),
        bounded,
    )
    .await
    .unwrap();
    let (control, events) = session.into_parts();
    for _ in 0..4 {
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text("old")))
            .unwrap();
    }
    control
        .send_with_fence(realtime::RealtimeInput::Text("input".into()), 7)
        .unwrap();
    let task = tokio::spawn(driver.run());
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while decoded.load(std::sync::atomic::Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("driver must reach the full event queue");
    tokio::task::yield_now().await;
    assert!(
        decoded.load(std::sync::atomic::Ordering::SeqCst) < 4,
        "the driver must be blocked while emitting queued old events"
    );
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        control.abort(RealtimeClose::normal("abort")),
    )
    .await
    .expect("abort must preempt a blocked event send")
    .unwrap();
    task.await.unwrap().unwrap();
    assert!(
        peer.sent.lock().unwrap().is_empty(),
        "fenced input must not be sent after abort"
    );
    let mut events = events.into_ordered();
    while let Some(event) = events.next().await {
        assert!(!matches!(event, RealtimeOrderedEvent::InputSent { .. }));
    }
}

#[test]
fn initialization_errors_and_late_oversized_frames_never_connect() {
    struct InitialCodec(Result<Vec<RealtimeFrame>, RealtimeError>);
    impl RealtimeCodec for InitialCodec {
        fn initial_frames(&self) -> Result<Vec<RealtimeFrame>, RealtimeError> {
            self.0.clone()
        }
        fn encode(&self, _: &realtime::RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
            unreachable!()
        }
        fn decode(&self, _: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
            unreachable!()
        }
    }
    for initial in [
        Err(RealtimeError::InvalidConfig {
            message: "invalid setup".into(),
        }),
        Ok(vec![
            RealtimeFrame::text("valid first frame"),
            RealtimeFrame::text(vec![b'x'; 4097]),
        ]),
    ] {
        let (transport, peer) = FakeTransport::new();
        let result = block_on(RealtimeSession::connect(
            &transport,
            request(),
            Arc::new(InitialCodec(initial)),
            limits(2, 4096),
        ));
        assert!(matches!(
            result,
            Err(RealtimeError::InvalidConfig { .. } | RealtimeError::FrameTooLarge { .. })
        ));
        assert!(
            transport.incoming.lock().unwrap().is_some(),
            "transport must not connect"
        );
        assert!(peer.sent.lock().unwrap().is_empty());
    }
}

#[test]
fn outbound_queue_is_bounded_and_drop_sends_a_normal_close() {
    block_on(async {
        let (control, _events, driver, peer) = connected(limits(1, 4096)).await;
        control
            .send(realtime::RealtimeInput::Text("hello".into()))
            .unwrap();
        assert_eq!(
            control.send(realtime::RealtimeInput::Text("next".into())),
            Err(RealtimeError::QueueFull)
        );
        drop(control);
        driver.run().await.unwrap();

        let sent = peer.sent.lock().unwrap();
        assert_eq!(sent.len(), 3, "setup frame and both text-turn frames");
        assert_eq!(
            *peer.closed.lock().unwrap(),
            Some(RealtimeClose::normal("realtime control dropped"))
        );
    });
}

#[test]
fn oversized_input_is_rejected_before_it_enters_the_queue() {
    block_on(async {
        let (control, _events, driver, _peer) = connected(limits(2, 4096)).await;
        let error = control
            .send(realtime::RealtimeInput::Audio {
                data: Bytes::from(vec![0; 4097]),
                format: realtime::RealtimeAudioFormat::Pcm16 {
                    sample_rate_hz: 24_000,
                },
            })
            .unwrap_err();
        assert!(matches!(
            error,
            RealtimeError::FrameTooLarge { max: 4096, .. }
        ));
        drop(control);
        driver.run().await.unwrap();
    });
}

#[test]
fn interrupt_is_encoded_as_openai_response_cancel() {
    block_on(async {
        let (control, _events, driver, peer) = connected(limits(2, 4096)).await;
        control.interrupt().unwrap();
        drop(control);
        driver.run().await.unwrap();

        let sent = peer.sent.lock().unwrap();
        let frames = sent
            .iter()
            .filter_map(|frame| match frame {
                RealtimeFrame::Text(bytes) => {
                    Some(serde_json::from_slice::<serde_json::Value>(bytes).unwrap())
                }
                RealtimeFrame::Binary(_) => None,
            })
            .collect::<Vec<_>>();
        assert!(frames
            .iter()
            .any(|frame| frame["type"] == "response.cancel"));
    });
}

#[test]
fn audio_delta_is_normalized_and_explicit_close_is_acknowledged() {
    block_on(async {
        let (control, mut events, driver, peer) = connected(limits(2, 4096)).await;
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"type":"response.output_audio.delta","item_id":"item-1","delta":"AQID"}"#,
            )))
            .unwrap();

        let session_task = async {
            let audio_event = events.next().await;
            let close_result = control.close(RealtimeClose::normal("done")).await;
            assert!(matches!(
                events.next().await,
                Some(RealtimeEvent::ProviderEvent { .. })
            ));
            let closed_event = events.next().await;
            (audio_event, close_result, closed_event)
        };
        let (session_result, driver_result) = futures::join!(session_task, driver.run());
        let (audio_event, close_result, closed_event) = session_result;
        close_result.unwrap();
        driver_result.unwrap();
        assert_eq!(
            audio_event,
            Some(RealtimeEvent::AudioDelta {
                data: Bytes::from_static(&[1, 2, 3]),
                format: realtime::RealtimeAudioFormat::Pcm16 {
                    sample_rate_hz: 24_000
                },
                item_id: Some("item-1".into())
            })
        );
        assert_eq!(
            closed_event,
            Some(RealtimeEvent::Closed {
                code: 1000,
                reason: "done".into()
            })
        );
        assert_eq!(
            *peer.closed.lock().unwrap(),
            Some(RealtimeClose::normal("done"))
        );
    });
}

#[test]
fn openai_codec_sets_the_session_and_decodes_tool_calls() {
    let codec = OpenAiRealtimeCodec::new(OpenAiRealtimeConfig {
        instructions: Some("Be concise".into()),
        output_mode: OpenAiOutputMode::Text,
        input_audio_format: OpenAiAudioFormat::G711MuLaw,
        output_audio_format: OpenAiAudioFormat::G711ALaw,
        voice: None,
        tools: vec![],
        tool_choice: Default::default(),
        ..Default::default()
    });
    let initial = codec.initial_frames().unwrap();
    assert_eq!(initial.len(), 1);
    let RealtimeFrame::Text(config) = &initial[0] else {
        panic!("OpenAI setup must be a JSON text frame");
    };
    let config: serde_json::Value = serde_json::from_slice(config).unwrap();
    assert_eq!(config["type"], "session.update");
    assert_eq!(config["session"]["instructions"], "Be concise");
    assert_eq!(config["session"]["output_modalities"][0], "text");
    assert_eq!(
        config["session"]["audio"]["input"]["format"]["type"],
        "audio/pcmu"
    );

    let events = codec
        .decode(RealtimeFrame::text(
            r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call-7","name":"lookup","arguments":"{\"q\":\"x\"}"}}"#,
        ))
        .unwrap();
    assert_eq!(
        events[..1],
        vec![RealtimeEvent::ToolCall {
            call_id: "call-7".into(),
            name: "lookup".into(),
            arguments: serde_json::json!({"q":"x"}),
        }]
    );
}

#[test]
fn openai_codec_maps_manual_audio_commit_and_tool_output() {
    let codec = OpenAiRealtimeCodec::default();
    let commit = codec.encode(&realtime::RealtimeInput::CommitAudio).unwrap();
    assert_eq!(commit.len(), 1);
    assert!(
        matches!(&commit[0], RealtimeFrame::Text(frame) if serde_json::from_slice::<serde_json::Value>(frame).unwrap()["type"] == "input_audio_buffer.commit")
    );
    let clear = codec.encode(&realtime::RealtimeInput::ClearAudio).unwrap();
    assert_eq!(clear.len(), 1);
    assert!(
        matches!(&clear[0], RealtimeFrame::Text(frame) if serde_json::from_slice::<serde_json::Value>(frame).unwrap()["type"] == "input_audio_buffer.clear")
    );
    assert!(codec
        .encode(&realtime::RealtimeInput::FinishSession)
        .is_err());
    for (input, expected_type) in [
        (
            realtime::RealtimeInput::RetrieveItem {
                item_id: "item-1".into(),
            },
            "conversation.item.retrieve",
        ),
        (
            realtime::RealtimeInput::DeleteItem {
                item_id: "item-1".into(),
            },
            "conversation.item.delete",
        ),
    ] {
        let frames = codec.encode(&input).unwrap();
        assert_eq!(frames.len(), 1);
        assert!(
            matches!(&frames[0], RealtimeFrame::Text(frame) if serde_json::from_slice::<serde_json::Value>(frame).unwrap()["type"] == expected_type)
        );
    }
    let truncate = codec
        .encode(&realtime::RealtimeInput::TruncateAudio {
            item_id: "assistant-item-1".into(),
            content_index: 0,
            audio_end_ms: 1_500,
        })
        .unwrap();
    assert!(matches!(
        &truncate[0],
        RealtimeFrame::Text(frame)
            if serde_json::from_slice::<serde_json::Value>(frame).unwrap()
                == serde_json::json!({
                    "type": "conversation.item.truncate",
                    "item_id": "assistant-item-1",
                    "content_index": 0,
                    "audio_end_ms": 1500,
                })
    ));
    assert!(codec
        .encode(&realtime::RealtimeInput::TruncateAudio {
            item_id: "assistant-item-1".into(),
            content_index: 1,
            audio_end_ms: 1_500,
        })
        .is_err());
    let tool_result = codec
        .encode(&realtime::RealtimeInput::ToolResult {
            call_id: "call-7".into(),
            output: serde_json::json!({"ok":true}),
        })
        .unwrap();
    assert_eq!(tool_result.len(), 1);
    assert!(
        matches!(&tool_result[0], RealtimeFrame::Text(frame) if serde_json::from_slice::<serde_json::Value>(frame).unwrap()["item"]["type"] == "function_call_output")
    );
    let continuation = codec
        .encode(&realtime::RealtimeInput::ContinueResponse)
        .unwrap();
    assert!(
        matches!(&continuation[0], RealtimeFrame::Text(frame) if serde_json::from_slice::<serde_json::Value>(frame).unwrap()["type"] == "response.create")
    );
    assert!(!tool_result[0].is_empty());
}

#[test]
fn codec_rejects_binary_frames_and_incompatible_audio() {
    let codec = OpenAiRealtimeCodec::default();
    assert!(codec
        .decode(RealtimeFrame::binary(Bytes::from_static(b"raw")))
        .is_err());
    assert!(codec
        .encode(&realtime::RealtimeInput::Audio {
            data: Bytes::from_static(b"data"),
            format: realtime::RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: 16_000,
            },
        })
        .is_err());
    assert!(codec
        .encode(&realtime::RealtimeInput::Audio {
            data: Bytes::from_static(b"data"),
            format: realtime::RealtimeAudioFormat::Encoded {
                mime_type: "audio/opus".into(),
            },
        })
        .is_err());
    let _provider_transport_error = RealtimeError::Transport {
        message: "fake transport failure".into(),
    };
}

#[test]
fn unexpected_remote_end_is_reported_as_interruption() {
    block_on(async {
        let (control, mut events, driver, peer) = connected(limits(2, 4096)).await;
        drop(peer.incoming);
        let result = driver.run().await;
        assert_eq!(result, Err(RealtimeError::UnexpectedRemoteClose));
        assert_eq!(
            events.next().await,
            Some(RealtimeEvent::ConnectionInterrupted {
                message: RealtimeError::UnexpectedRemoteClose.to_string()
            })
        );
        drop(control);
    });
}

#[test]
fn connect_request_debug_redacts_endpoint_and_header_values() {
    let debug = format!("{:?}", request());
    assert!(!debug.contains("secret"));
    assert!(debug.contains("Authorization"));
}

#[test]
fn abort_discards_queued_audio_before_driver_runs() {
    block_on(async {
        let (control, mut events, driver, peer) = connected(limits(2, 4096)).await;
        for _ in 0..2 {
            control
                .send(realtime::RealtimeInput::Audio {
                    data: Bytes::from_static(&[0, 0]),
                    format: realtime::RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 24000,
                    },
                })
                .unwrap();
        }
        let abort = control.abort(RealtimeClose::normal("cancel"));
        let (aborted, driven) = futures::join!(abort, driver.run());
        aborted.unwrap();
        driven.unwrap();
        assert_eq!(
            peer.sent.lock().unwrap().len(),
            1,
            "only setup sent; queued audio discarded"
        );
        assert!(matches!(
            control.send(realtime::RealtimeInput::Text("late".into())),
            Err(RealtimeError::Closed)
        ));
        assert!(matches!(
            events.next().await,
            Some(RealtimeEvent::Closed { .. })
        ));
    });
}

#[tokio::test]
async fn abort_preempts_pending_transport_send_and_full_event_queue() {
    struct StallTransport {
        closed: Arc<std::sync::atomic::AtomicBool>,
    }
    struct StallSink {
        first: bool,
        closed: Arc<std::sync::atomic::AtomicBool>,
    }
    #[async_trait]
    impl RealtimeSink for StallSink {
        async fn ping(&mut self, _payload: bytes::Bytes) -> Result<(), RealtimeError> {
            Err(RealtimeError::InvalidInput {
                message: "test transport does not support explicit WebSocket Ping frames".into(),
            })
        }

        async fn send(&mut self, _: RealtimeFrame) -> Result<(), RealtimeError> {
            if self.first {
                self.first = false;
                return Ok(());
            }
            futures::future::pending().await
        }
        fn abort(&mut self) {
            self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        async fn close(&mut self, _: RealtimeClose) -> Result<(), RealtimeError> {
            panic!("abort must not flush graceful close");
        }
    }
    #[async_trait]
    impl RealtimeTransport for StallTransport {
        async fn connect(
            &self,
            _: RealtimeConnectRequest,
        ) -> Result<RealtimeConnection, RealtimeError> {
            Ok(RealtimeConnection {
                outbound: Box::new(StallSink {
                    first: true,
                    closed: self.closed.clone(),
                }),
                inbound: futures::stream::pending().boxed(),
            })
        }
    }
    let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (session, driver) = RealtimeSession::connect(
        &StallTransport {
            closed: closed.clone(),
        },
        request(),
        Arc::new(OpenAiRealtimeCodec::default()),
        limits(2, 4096),
    )
    .await
    .unwrap();
    let (control, _events) = session.into_parts();
    control
        .send(realtime::RealtimeInput::Audio {
            data: Bytes::from_static(&[0, 0]),
            format: realtime::RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: 24000,
            },
        })
        .unwrap();
    let pump = tokio::spawn(driver.run());
    tokio::task::yield_now().await;
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        control.abort(RealtimeClose::normal("cancel")),
    )
    .await
    .unwrap()
    .unwrap();
    pump.await.unwrap().unwrap();
    assert!(closed.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test]
async fn abort_preempts_full_event_queue_without_waiting_for_receiver() {
    let (transport, peer) = FakeTransport::new();
    let mut bounds = limits(2, 4096);
    bounds.event_capacity = 1;
    let (session, driver) = RealtimeSession::connect(
        &transport,
        request(),
        Arc::new(OpenAiRealtimeCodec::default()),
        bounds,
    )
    .await
    .unwrap();
    let (control, mut events) = session.into_parts();
    peer.incoming
        .unbounded_send(Ok(RealtimeFrame::text(
            r#"{"type":"response.output_audio.delta","item_id":"a1","delta":"AAA="}"#,
        )))
        .unwrap();
    let pump = tokio::spawn(driver.run());
    tokio::task::yield_now().await;
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        control.abort(RealtimeClose::normal("cancel")),
    )
    .await
    .unwrap()
    .unwrap();
    pump.await.unwrap().unwrap();
    assert!(matches!(
        events.next().await,
        Some(RealtimeEvent::AudioDelta { .. })
    ));
    assert_eq!(
        events.next().await,
        None,
        "termination must not await room for a Closed event"
    );
}
