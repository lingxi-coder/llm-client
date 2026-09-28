use async_trait::async_trait;
use futures::{
    channel::{mpsc, oneshot},
    executor::{block_on, LocalPool},
    stream::{self, BoxStream},
    task::LocalSpawnExt,
    StreamExt,
};
use lingxi_llm_client::{
    protocol::Secret,
    providers::xai::audio::{XaiSpeechCodec, XaiSpeechFormat},
    providers::xai::streaming_tts::{
        XaiStreamingTtsConfig, XaiStreamingTtsEvent, XaiStreamingTtsSession,
        XAI_STREAMING_TTS_WEBSOCKET_ENDPOINT,
    },
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeLimits, RealtimeSink, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use url::Url;

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    sent_frames: mpsc::UnboundedSender<RealtimeFrame>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
    send_error: Arc<Mutex<Option<(String, RealtimeError)>>>,
}

struct FakePeer {
    incoming: Option<mpsc::UnboundedSender<Incoming>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    sent_frames: mpsc::UnboundedReceiver<RealtimeFrame>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    sent_frames: mpsc::UnboundedSender<RealtimeFrame>,
    send_error: Arc<Mutex<Option<(String, RealtimeError)>>>,
}

fn fake_transport(
    preface: &[Value],
    send_error: Option<(String, RealtimeError)>,
) -> (Arc<FakeTransport>, FakePeer) {
    let (incoming_tx, incoming_rx) = mpsc::unbounded();
    let prefix = preface
        .iter()
        .map(|value| Ok(RealtimeFrame::text(value.to_string())))
        .collect::<Vec<_>>();
    let incoming = stream::iter(prefix).chain(incoming_rx).boxed();
    let (sent_frames_tx, sent_frames_rx) = mpsc::unbounded();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let request = Arc::new(Mutex::new(None));
    let connect_count = Arc::new(Mutex::new(0));
    let send_error = Arc::new(Mutex::new(send_error));
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(incoming)),
            sent: sent.clone(),
            sent_frames: sent_frames_tx,
            request: request.clone(),
            connect_count: connect_count.clone(),
            send_error,
        }),
        FakePeer {
            incoming: Some(incoming_tx),
            sent,
            sent_frames: sent_frames_rx,
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
            .expect("fake accepts one connection");
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
                sent_frames: self.sent_frames.clone(),
                send_error: self.send_error.clone(),
            }),
            inbound: incoming,
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        let event_type = frame_type(&frame);
        let send_error = {
            let mut send_error = self.send_error.lock().unwrap();
            if send_error.as_ref().is_some_and(|(expected_type, _)| {
                event_type.as_deref() == Some(expected_type.as_str())
            }) {
                send_error.take().map(|(_, error)| error)
            } else {
                None
            }
        };
        if let Some(error) = send_error {
            return Err(error);
        }
        self.sent.lock().unwrap().push(frame.clone());
        let _ = self.sent_frames.unbounded_send(frame);
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}

fn frame_type(frame: &RealtimeFrame) -> Option<String> {
    match frame {
        RealtimeFrame::Text(bytes) => serde_json::from_slice::<Value>(bytes)
            .ok()?
            .get("type")?
            .as_str()
            .map(str::to_owned),
        RealtimeFrame::Binary(_) => None,
    }
}

fn ready_limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 16,
        max_frame_bytes: 256 * 1024,
    }
}

fn connect(
    fake: Arc<FakeTransport>,
    config: XaiStreamingTtsConfig,
    limits: RealtimeLimits,
) -> Result<
    (
        lingxi_llm_client::providers::xai::streaming_tts::XaiStreamingTtsSession,
        lingxi_llm_client::providers::xai::streaming_tts::XaiStreamingTtsDriver,
    ),
    lingxi_llm_client::providers::xai::streaming_tts::XaiStreamingTtsError,
> {
    block_on(XaiStreamingTtsSession::connect(
        fake,
        Secret::from("ephemeral-xai-key".to_owned()),
        config,
        limits,
    ))
}

fn spawn_driver(
    pool: &LocalPool,
    driver: lingxi_llm_client::providers::xai::streaming_tts::XaiStreamingTtsDriver,
) -> oneshot::Receiver<Result<(), RealtimeError>> {
    let (send, receive) = oneshot::channel();
    pool.spawner()
        .spawn_local(async move {
            let _ = send.send(driver.run().await);
        })
        .unwrap();
    receive
}

fn next_sent(pool: &mut LocalPool, peer: &mut FakePeer) -> RealtimeFrame {
    pool.run_until(peer.sent_frames.next())
        .expect("driver sends the expected frame")
}

fn next_event(
    pool: &mut LocalPool,
    events: &mut lingxi_llm_client::providers::xai::streaming_tts::XaiStreamingTtsEvents,
) -> XaiStreamingTtsEvent {
    pool.run_until(events.next())
        .expect("driver emits the expected event")
}

fn push_event(peer: &FakePeer, value: Value) {
    peer.incoming
        .as_ref()
        .expect("fake peer is still connected")
        .unbounded_send(Ok(RealtimeFrame::text(value.to_string())))
        .unwrap();
}

fn text_json(frame: RealtimeFrame) -> Value {
    match frame {
        RealtimeFrame::Text(bytes) => serde_json::from_slice(&bytes).unwrap(),
        RealtimeFrame::Binary(_) => panic!("TTS client messages use JSON text frames"),
    }
}

#[test]
fn uses_fixed_streaming_route_bearer_auth_and_documented_query_options() {
    let (fake, peer) = fake_transport(&[], None);
    let mut config = XaiStreamingTtsConfig::new("pt-BR");
    config.voice = Some("eve".into());
    config.output_format = XaiSpeechFormat {
        codec: XaiSpeechCodec::Mp3,
        sample_rate: 44_100,
        bit_rate: Some(192_000),
    };
    config.speed = 1.2;
    config.optimize_streaming_latency = 2;
    config.text_normalization = true;
    config.with_timestamps = true;
    let (_session, _driver) = connect(fake, config, ready_limits()).unwrap();

    let request = peer.request.lock().unwrap().clone().unwrap();
    assert_eq!(
        request.endpoint.split('?').next(),
        Some(XAI_STREAMING_TTS_WEBSOCKET_ENDPOINT)
    );
    assert_eq!(request.headers.len(), 1);
    assert_eq!(request.headers[0].0, "authorization");
    assert_eq!(request.headers[0].1, "Bearer ephemeral-xai-key");
    let endpoint = Url::parse(&request.endpoint).unwrap();
    assert_eq!(endpoint.scheme(), "wss");
    assert_eq!(endpoint.host_str(), Some("api.x.ai"));
    assert_eq!(endpoint.path(), "/v1/tts");
    let query = endpoint.query_pairs().collect::<Vec<_>>();
    for (key, value) in [
        ("language", "pt-BR"),
        ("voice", "eve"),
        ("codec", "mp3"),
        ("sample_rate", "44100"),
        ("bit_rate", "192000"),
        ("speed", "1.2"),
        ("optimize_streaming_latency", "2"),
        ("text_normalization", "true"),
        ("with_timestamps", "true"),
    ] {
        assert!(query
            .iter()
            .any(|(actual_key, actual_value)| { actual_key == key && actual_value == value }));
    }
    assert_eq!(*peer.connect_count.lock().unwrap(), 1);
}

#[test]
fn invalid_config_is_rejected_before_the_websocket_handshake() {
    let (fake, peer) = fake_transport(&[], None);
    let mut config = XaiStreamingTtsConfig::new("en");
    config.optimize_streaming_latency = 3;
    let result = connect(fake, config, ready_limits());
    assert!(matches!(
        result,
        Err(
            lingxi_llm_client::providers::xai::streaming_tts::XaiStreamingTtsError::Realtime(
                RealtimeError::InvalidConfig { .. }
            )
        )
    ));
    assert_eq!(*peer.connect_count.lock().unwrap(), 0);

    let (fake, peer) = fake_transport(&[], None);
    let config = XaiStreamingTtsConfig::new(" \n ");
    assert!(connect(fake, config, ready_limits()).is_err());
    assert_eq!(*peer.connect_count.lock().unwrap(), 0);
}

#[test]
fn update_map_is_a_native_session_update_and_stream_supports_multiple_turns() {
    let (fake, mut peer) = fake_transport(&[], None);
    let (session, driver) =
        connect(fake, XaiStreamingTtsConfig::new("en"), ready_limits()).unwrap();
    let (control, mut events) = session.into_parts();
    let replacements = BTreeMap::from([("Acme Mobile".into(), "Acme Mobull".into())]);
    control.update_replacements(&replacements).unwrap();
    control.send_text_delta("Acme Mobile is here.").unwrap();
    control.finish_utterance().unwrap();

    let mut pool = LocalPool::new();
    let driver_result = spawn_driver(&pool, driver);
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"session.update","replace":{"Acme Mobile":"Acme Mobull"}})
    );
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.delta","delta":"Acme Mobile is here."})
    );
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.done"})
    );

    push_event(
        &peer,
        json!({"type":"session.updated","replace":{"Acme Mobile":"Acme Mobull"}}),
    );
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::SessionUpdated { replace, .. }
            if replace["Acme Mobile"] == "Acme Mobull"
    ));
    push_event(
        &peer,
        json!({
            "type":"audio.delta",
            "delta":"AQID",
            "audio_duration":0.1,
            "audio_timestamps":{"graph_chars":["A"],"graph_times":[[0.0,0.1]]}
        }),
    );
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::AudioDelta { audio, native }
            if audio.as_ref() == [1, 2, 3] && native["audio_timestamps"]["graph_chars"][0] == "A"
    ));
    push_event(&peer, json!({"type":"audio.done","trace_id":"trace-one"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::AudioDone { trace_id, .. } if trace_id == "trace-one"
    ));

    control.send_text_delta("Turn two.").unwrap();
    control.finish_utterance().unwrap();
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.delta","delta":"Turn two."})
    );
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.done"})
    );
    push_event(&peer, json!({"type":"audio.done","trace_id":"trace-two"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::AudioDone { trace_id, .. } if trace_id == "trace-two"
    ));

    pool.run_until(control.close()).unwrap();
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::LocalClose { code: 1000, .. }
    ));
    assert_eq!(pool.run_until(driver_result).unwrap(), Ok(()));
}

#[test]
fn text_clear_blocks_new_text_until_ack_and_retires_the_cancelled_done_generation() {
    let (fake, mut peer) = fake_transport(&[], None);
    let (session, driver) =
        connect(fake, XaiStreamingTtsConfig::new("en"), ready_limits()).unwrap();
    let (control, mut events) = session.into_parts();
    let mut pool = LocalPool::new();
    let driver_result = spawn_driver(&pool, driver);

    control.send_text_delta("Cancel this turn.").unwrap();
    control.finish_utterance().unwrap();
    let _ = next_sent(&mut pool, &mut peer);
    let _ = next_sent(&mut pool, &mut peer);
    control.clear_utterance().unwrap();
    assert!(matches!(
        control.send_text_delta("too early"),
        Err(RealtimeError::InvalidInput { .. })
    ));
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.clear"})
    );

    push_event(&peer, json!({"type":"audio.delta","delta":"AQI="}));
    push_event(&peer, json!({"type":"audio.clear"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::AudioDelta { audio, .. } if audio.as_ref() == [1, 2]
    ));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::AudioClear { .. }
    ));

    control.send_text_delta("New turn.").unwrap();
    let _ = next_sent(&mut pool, &mut peer);
    // This completion is deliberately readable before the queued text.done is
    // sent. It belongs to the canceled turn and must not advance the new one.
    push_event(&peer, json!({"type":"transcript.future","kept":true}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ProviderEvent { name, .. } if name == "transcript.future"
    ));
    push_event(&peer, json!({"type":"transcript.future","kept":true}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ProviderEvent { name, .. } if name == "transcript.future"
    ));
    control.finish_utterance().unwrap();
    push_event(&peer, json!({"type":"audio.done","trace_id":"stale"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ProviderEvent { name, .. } if name == "audio.done"
    ));
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.done"})
    );
    push_event(&peer, json!({"type":"audio.done","trace_id":"fresh"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::AudioDone { trace_id, .. } if trace_id == "fresh"
    ));

    pool.run_until(control.close()).unwrap();
    assert_eq!(pool.run_until(driver_result).unwrap(), Ok(()));
}

#[test]
fn outbound_queue_is_bounded_and_queue_full_does_not_advance_turn_state() {
    let (fake, mut peer) = fake_transport(&[], None);
    let (session, driver) = connect(
        fake,
        XaiStreamingTtsConfig::new("en"),
        RealtimeLimits {
            outbound_capacity: 1,
            ..ready_limits()
        },
    )
    .unwrap();
    let (control, _events) = session.into_parts();
    control.send_text_delta("one").unwrap();
    assert_eq!(control.finish_utterance(), Err(RealtimeError::QueueFull));

    let mut pool = LocalPool::new();
    let driver_result = spawn_driver(&pool, driver);
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.delta","delta":"one"})
    );
    control.finish_utterance().unwrap();
    assert_eq!(
        text_json(next_sent(&mut pool, &mut peer)),
        json!({"type":"text.done"})
    );
    pool.run_until(control.close()).unwrap();
    assert_eq!(pool.run_until(driver_result).unwrap(), Ok(()));
}

#[test]
fn an_early_audio_done_cannot_complete_before_text_done_reaches_the_socket() {
    let (fake, mut peer) = fake_transport(
        &[],
        Some((
            "text.done".into(),
            RealtimeError::Transport {
                message: "write failed".into(),
            },
        )),
    );
    let (session, driver) =
        connect(fake, XaiStreamingTtsConfig::new("en"), ready_limits()).unwrap();
    let (control, mut events) = session.into_parts();
    control.send_text_delta("Hello.").unwrap();
    let mut pool = LocalPool::new();
    let driver_result = spawn_driver(&pool, driver);
    assert!(matches!(
        text_json(next_sent(&mut pool, &mut peer))["type"].as_str(),
        Some("text.delta")
    ));
    push_event(&peer, json!({"type":"tts.future"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ProviderEvent { name, .. } if name == "tts.future"
    ));
    control.finish_utterance().unwrap();
    push_event(&peer, json!({"type":"audio.done","trace_id":"too-early"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ProviderEvent { name, .. } if name == "audio.done"
    ));
    assert!(matches!(
        pool.run_until(driver_result).unwrap(),
        Err(RealtimeError::Transport { .. })
    ));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ConnectionInterrupted { .. }
    ));
}

#[test]
fn malformed_and_unknown_events_are_retained_and_remote_eof_is_an_error() {
    let (fake, mut peer) = fake_transport(&[], None);
    let (session, driver) =
        connect(fake, XaiStreamingTtsConfig::new("en"), ready_limits()).unwrap();
    let (control, mut events) = session.into_parts();
    control.send_text_delta("Say a line.").unwrap();
    control.finish_utterance().unwrap();
    let mut pool = LocalPool::new();
    let driver_result = spawn_driver(&pool, driver);
    let _ = next_sent(&mut pool, &mut peer);
    let _ = next_sent(&mut pool, &mut peer);

    push_event(&peer, json!({"type":"audio.delta","delta":"!bad base64!"}));
    let malformed = next_event(&mut pool, &mut events);
    assert!(matches!(malformed,
        XaiStreamingTtsEvent::ProviderEvent { name, native }
            if name == "audio.delta" && native["delta"] == "!bad base64!"));
    let unknown = json!({"type":"audio.future","native":{"kept":true}});
    push_event(&peer, unknown.clone());
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ProviderEvent { name, native }
            if name == "audio.future" && native == unknown
    ));
    push_event(&peer, json!({"type":"error","message":"synthesis failed"}));
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ProviderError { message, .. } if message == "synthesis failed"
    ));

    peer.finish();
    assert_eq!(
        pool.run_until(driver_result).unwrap(),
        Err(RealtimeError::UnexpectedRemoteClose)
    );
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::ConnectionInterrupted { .. }
    ));
}

#[test]
fn dropping_the_last_control_closes_the_socket_and_finishes_the_driver() {
    let (fake, peer) = fake_transport(&[], None);
    let (session, driver) =
        connect(fake, XaiStreamingTtsConfig::new("en"), ready_limits()).unwrap();
    let (control, mut events) = session.into_parts();
    let mut pool = LocalPool::new();
    let driver_result = spawn_driver(&pool, driver);
    drop(control);
    assert!(matches!(
        next_event(&mut pool, &mut events),
        XaiStreamingTtsEvent::LocalClose { code: 1000, .. }
    ));
    assert_eq!(pool.run_until(driver_result).unwrap(), Ok(()));
    assert!(peer.sent.lock().unwrap().is_empty());
}

impl FakePeer {
    fn finish(&mut self) {
        self.incoming.take();
    }
}
