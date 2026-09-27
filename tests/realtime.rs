use async_trait::async_trait;
use bytes::Bytes;
use futures::{channel::mpsc, executor::block_on, stream::BoxStream, StreamExt};
use lingxi_llm_client::realtime;
use lingxi_llm_client::realtime::{
    OpenAiAudioFormat, OpenAiOutputMode, OpenAiRealtimeCodec, OpenAiRealtimeConfig, RealtimeClose,
    RealtimeCodec, RealtimeConnectRequest, RealtimeConnection, RealtimeControl, RealtimeDriver,
    RealtimeError, RealtimeEvent, RealtimeFrame, RealtimeLimits, RealtimeSession, RealtimeSink,
    RealtimeTransport,
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
        .expect_err("the default sink method must report unsupported Ping capability");

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
        events,
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
    assert_eq!(tool_result.len(), 2);
    assert!(
        matches!(&tool_result[0], RealtimeFrame::Text(frame) if serde_json::from_slice::<serde_json::Value>(frame).unwrap()["item"]["type"] == "function_call_output")
    );
    assert!(
        matches!(&tool_result[1], RealtimeFrame::Text(frame) if serde_json::from_slice::<serde_json::Value>(frame).unwrap()["type"] == "response.create")
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
