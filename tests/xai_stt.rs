use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::mpsc,
    executor::block_on,
    future::{join, poll_fn},
    stream::{self, BoxStream},
    StreamExt,
};
use lingxi_llm_client::{
    protocol::Secret,
    providers::xai::stt::{
        XaiSttConfig, XaiSttEncoding, XaiSttEvent, XaiSttModel, XaiSttSession,
        XAI_STT_WEBSOCKET_ENDPOINT,
    },
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeLimits, RealtimeSink, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::Poll,
    time::Duration,
};
use url::Url;

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
    send_error: Arc<Mutex<Option<RealtimeError>>>,
}

struct FakePeer {
    incoming: Option<mpsc::UnboundedSender<Incoming>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    send_error: Arc<Mutex<Option<RealtimeError>>>,
}

fn fake_transport(preface: &[Value]) -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_send_error(preface, None)
}

fn fake_transport_with_send_error(
    preface: &[Value],
    send_error: Option<RealtimeError>,
) -> (Arc<FakeTransport>, FakePeer) {
    let (incoming_tx, incoming_rx) = mpsc::unbounded();
    let prefix = preface
        .iter()
        .map(|value| Ok(RealtimeFrame::text(value.to_string())))
        .collect::<Vec<_>>();
    let incoming = stream::iter(prefix).chain(incoming_rx).boxed();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let request = Arc::new(Mutex::new(None));
    let connect_count = Arc::new(Mutex::new(0));
    let send_error = Arc::new(Mutex::new(send_error));
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(incoming)),
            sent: sent.clone(),
            request: request.clone(),
            connect_count: connect_count.clone(),
            send_error,
        }),
        FakePeer {
            incoming: Some(incoming_tx),
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
            .expect("fake accepts one connection");
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
                send_error: self.send_error.clone(),
            }),
            inbound: incoming,
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        let is_audio_done = matches!(&frame,
            RealtimeFrame::Text(bytes)
                if serde_json::from_slice::<Value>(bytes)
                    .ok()
                    .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned))
                    .as_deref() == Some("audio.done"));
        if is_audio_done {
            if let Some(error) = self.send_error.lock().unwrap().take() {
                return Err(error);
            }
        }
        self.sent.lock().unwrap().push(frame);
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}

fn ready() -> Value {
    json!({"type":"transcript.created"})
}

fn limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 16,
        max_frame_bytes: 64 * 1024,
    }
}

fn credential() -> Secret<String> {
    Secret::from("ephemeral-xai-key".to_owned())
}

async fn progress_until_sent<F>(
    driver: &mut Pin<Box<F>>,
    sent: &Arc<Mutex<Vec<RealtimeFrame>>>,
    expected: usize,
) where
    F: Future<Output = Result<(), RealtimeError>>,
{
    let sent = sent.clone();
    poll_fn(|cx| {
        let result = driver.as_mut().poll(cx);
        if sent.lock().unwrap().len() >= expected {
            Poll::Ready(())
        } else if let Poll::Ready(result) = result {
            panic!("driver ended before sending queued frames: {result:?}");
        } else {
            Poll::Pending
        }
    })
    .await;
}

async fn drain_events(
    mut events: lingxi_llm_client::providers::xai::stt::XaiSttEvents,
) -> Vec<XaiSttEvent> {
    let mut received = Vec::new();
    while let Some(event) = events.next().await {
        received.push(event);
    }
    received
}

#[test]
fn uses_fixed_official_route_query_and_per_connection_bearer_auth() {
    block_on(async {
        let (fake, peer) = fake_transport(&[ready()]);
        let config = XaiSttConfig {
            model: XaiSttModel::GrokVoiceTranscribe1,
            encoding: XaiSttEncoding::Pcm,
            sample_rate_hz: Some(24_000),
            interim_results: true,
            endpointing_ms: 250,
            language: Some("en".into()),
            diarize: Some(true),
            filler_words: true,
            multichannel: true,
            channels: 2,
            keyterms: vec!["Grok".into(), "xAI".into()],
            smart_turn: Some(0.7),
            smart_turn_timeout_ms: Some(2_500),
            vad_threshold: Some(0.1),
            ..XaiSttConfig::default()
        };
        let (_session, _driver) = XaiSttSession::connect(fake, credential(), config, limits())
            .await
            .unwrap();

        let request = peer.request.lock().unwrap().clone().unwrap();
        assert_eq!(
            request.endpoint.split('?').next(),
            Some(XAI_STT_WEBSOCKET_ENDPOINT)
        );
        let url = Url::parse(&request.endpoint).unwrap();
        let query = url.query_pairs().collect::<Vec<_>>();
        assert_eq!(url.scheme(), "wss");
        assert_eq!(url.host_str(), Some("api.x.ai"));
        assert_eq!(url.path(), "/v1/stt");
        for (key, value) in [
            ("model", "grok-voice-transcribe-1.0"),
            ("encoding", "pcm"),
            ("sample_rate", "24000"),
            ("interim_results", "true"),
            ("endpointing", "250"),
            ("language", "en"),
            ("diarize", "true"),
            ("filler_words", "true"),
            ("multichannel", "true"),
            ("channels", "2"),
            ("smart_turn", "0.7"),
            ("smart_turn_timeout", "2500"),
            ("vad_threshold", "0.1"),
        ] {
            assert!(query
                .iter()
                .any(|(actual_key, actual_value)| { actual_key == key && actual_value == value }));
        }
        assert_eq!(
            query
                .iter()
                .filter(|(key, _)| key == "keyterm")
                .map(|(_, value)| value.as_ref())
                .collect::<Vec<_>>(),
            ["Grok", "xAI"]
        );
        assert_eq!(request.headers.len(), 1);
        assert_eq!(request.headers[0].0, "authorization");
        assert_eq!(request.headers[0].1, "Bearer ephemeral-xai-key");
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
    });
}

#[test]
fn opus_omits_sample_rate_and_rejects_opus_multichannel_before_transport() {
    block_on(async {
        let (fake, peer) = fake_transport(&[ready()]);
        let config = XaiSttConfig {
            encoding: XaiSttEncoding::Opus,
            interim_results: true,
            ..XaiSttConfig::default()
        };
        let (session, _driver) = XaiSttSession::connect(fake, credential(), config, limits())
            .await
            .unwrap();
        let request = peer.request.lock().unwrap().clone().unwrap();
        let url = Url::parse(&request.endpoint).unwrap();
        assert_eq!(
            url.query_pairs()
                .find(|(key, _)| key == "encoding")
                .unwrap()
                .1,
            "opus"
        );
        assert!(!url.query_pairs().any(|(key, _)| key == "sample_rate"));
        let (control, _) = session.into_parts();
        assert!(control
            .send_audio(Bytes::from_static(b"one-opus-packet"))
            .is_ok());

        let (fake, peer) = fake_transport(&[]);
        let invalid = XaiSttConfig {
            encoding: XaiSttEncoding::Opus,
            multichannel: true,
            channels: 2,
            ..XaiSttConfig::default()
        };
        let result = XaiSttSession::connect(fake, credential(), invalid, limits()).await;
        assert!(matches!(
            result,
            Err(
                lingxi_llm_client::providers::xai::stt::XaiSttError::Realtime(
                    RealtimeError::InvalidConfig { .. }
                )
            )
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
    });
}

#[test]
fn queues_raw_audio_finalize_and_audio_done_in_clone_order() {
    block_on(async {
        let (fake, peer) = fake_transport(&[ready()]);
        let (session, driver) =
            XaiSttSession::connect(fake, credential(), XaiSttConfig::default(), limits())
                .await
                .unwrap();
        let (control, _) = session.into_parts();
        let clone = control.clone();
        control.send_audio(Bytes::from_static(&[1, 2, 3])).unwrap();
        clone.finalize_utterance().unwrap();
        control.finish_audio().unwrap();
        assert!(matches!(
            control.send_audio(Bytes::from_static(&[4])),
            Err(RealtimeError::Closed)
        ));

        let mut driver = Box::pin(driver.run());
        progress_until_sent(&mut driver, &peer.sent, 3).await;
        let frames = peer.sent.lock().unwrap().clone();
        assert!(matches!(&frames[0], RealtimeFrame::Binary(bytes) if bytes.as_ref() == [1, 2, 3]));
        assert!(
            matches!(&frames[1], RealtimeFrame::Text(bytes) if serde_json::from_slice::<Value>(bytes).unwrap() == json!({"type":"finalize"}))
        );
        assert!(
            matches!(&frames[2], RealtimeFrame::Text(bytes) if serde_json::from_slice::<Value>(bytes).unwrap() == json!({"type":"audio.done"}))
        );
    });
}

#[test]
fn queue_full_does_not_permanently_close_audio_input() {
    block_on(async {
        let (fake, _peer) = fake_transport(&[ready()]);
        let tiny_limits = RealtimeLimits {
            outbound_capacity: 1,
            ..limits()
        };
        let (session, _driver) =
            XaiSttSession::connect(fake, credential(), XaiSttConfig::default(), tiny_limits)
                .await
                .unwrap();
        let (control, _) = session.into_parts();
        control.send_audio(Bytes::from_static(b"queued")).unwrap();
        assert_eq!(control.finish_audio(), Err(RealtimeError::QueueFull));
        assert_eq!(
            control.send_audio(Bytes::from_static(b"still open")),
            Err(RealtimeError::QueueFull)
        );
    });
}

#[test]
fn valid_mono_final_requires_sent_audio_done_and_real_remote_eof() {
    block_on(async {
        let (fake, mut peer) = fake_transport(&[ready()]);
        let (session, driver) =
            XaiSttSession::connect(fake, credential(), XaiSttConfig::default(), limits())
                .await
                .unwrap();
        let (control, events) = session.into_parts();
        control.finish_audio().unwrap();
        let mut driver = Box::pin(driver.run());
        progress_until_sent(&mut driver, &peer.sent, 1).await;
        peer.push(json!({"type":"transcript.done","text":"hello","duration":1.25}));
        peer.finish();
        let (driver_result, events) = join(driver, drain_events(events)).await;
        assert_eq!(driver_result, Ok(()));
        assert!(events.iter().any(|event| matches!(event,
            XaiSttEvent::TranscriptDone { duration, .. } if *duration == 1.25)));
        assert!(events.iter().any(|event| matches!(
            event,
            XaiSttEvent::Completed {
                expected_channels: 1
            }
        )));
    });
}

#[test]
fn multichannel_done_is_validated_deduplicated_and_waits_for_each_channel() {
    block_on(async {
        let config = XaiSttConfig {
            multichannel: true,
            channels: 2,
            ..XaiSttConfig::default()
        };
        let (fake, mut peer) = fake_transport(&[ready()]);
        let (session, driver) = XaiSttSession::connect(fake, credential(), config, limits())
            .await
            .unwrap();
        let (control, events) = session.into_parts();
        control.finish_audio().unwrap();
        let mut driver = Box::pin(driver.run());
        progress_until_sent(&mut driver, &peer.sent, 1).await;
        peer.push(json!({"type":"transcript.done","channel_index":0,"duration":1.0,"text":"a"}));
        peer.push(
            json!({"type":"transcript.done","channel_index":0,"duration":1.0,"text":"duplicate"}),
        );
        peer.push(json!({"type":"transcript.done","channel_index":2,"duration":1.0,"text":"out of range"}));
        peer.push(json!({"type":"transcript.done","channel_index":1,"duration":2.0,"text":"b"}));
        peer.finish();
        let (driver_result, events) = join(driver, drain_events(events)).await;
        assert_eq!(driver_result, Ok(()));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, XaiSttEvent::TranscriptDone { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, XaiSttEvent::ProviderEvent { name, .. } if name == "transcript.done"))
                .count(),
            2
        );
        assert!(events.iter().any(|event| matches!(
            event,
            XaiSttEvent::Completed {
                expected_channels: 2
            }
        )));
    });
}

#[test]
fn partials_are_typed_only_when_documented_fields_are_valid_and_unknown_events_survive() {
    block_on(async {
        let (fake, mut peer) = fake_transport(&[ready()]);
        let (session, driver) = XaiSttSession::connect(
            fake,
            credential(),
            XaiSttConfig {
                interim_results: true,
                ..XaiSttConfig::default()
            },
            limits(),
        )
        .await
        .unwrap();
        let (session_control, events) = session.into_parts();
        let driver = Box::pin(driver.run());
        peer.push(json!({
            "type":"transcript.partial",
            "text":"hello",
            "words":[{"text":"hello","start":0.0,"end":0.3}],
            "is_final":false,
            "speech_final":false,
            "start":0.0,
            "duration":0.3
        }));
        peer.push(json!({
            "type":"transcript.partial",
            "text":"malformed timestamp",
            "words":[],
            "is_final":false,
            "speech_final":false,
            "start":"zero",
            "duration":0.3
        }));
        let unknown = json!({"type":"transcript.future","payload":{"kept":true}});
        peer.push(unknown.clone());
        peer.finish();
        let (driver_result, events) = join(driver, drain_events(events)).await;
        assert_eq!(driver_result, Err(RealtimeError::UnexpectedRemoteClose));
        assert!(events.iter().any(|event| matches!(event,
            XaiSttEvent::TranscriptPartial { text, duration, .. } if text == "hello" && *duration == 0.3)));
        assert!(events.iter().any(|event| matches!(event,
            XaiSttEvent::ProviderEvent { name, native } if name == "transcript.partial" && native["start"] == "zero")));
        assert!(events.iter().any(|event| matches!(event,
            XaiSttEvent::ProviderEvent { name, native } if name == "transcript.future" && *native == unknown)));
        drop(session_control);
    });
}

#[test]
fn provider_error_before_ready_is_a_connect_error_with_native_payload() {
    block_on(async {
        let error = json!({"type":"error","message":"bad model","code":"invalid_model"});
        let (fake, peer) = fake_transport(std::slice::from_ref(&error));
        let result =
            XaiSttSession::connect(fake, credential(), XaiSttConfig::default(), limits()).await;
        assert!(matches!(
            result,
            Err(lingxi_llm_client::providers::xai::stt::XaiSttError::Provider { message, native })
                if message == "bad model" && native == error
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
    });
}

#[test]
fn malformed_or_pre_audio_done_is_preserved_and_eof_remains_failure() {
    block_on(async {
        let (fake, mut peer) = fake_transport(&[
            json!({"type":"transcript.done","duration":"1.0","text":"too early"}),
            ready(),
        ]);
        let (session, driver) =
            XaiSttSession::connect(fake, credential(), XaiSttConfig::default(), limits())
                .await
                .unwrap();
        let (control, events) = session.into_parts();
        control.finish_audio().unwrap();
        let mut driver = Box::pin(driver.run());
        progress_until_sent(&mut driver, &peer.sent, 1).await;
        peer.push(json!({"type":"transcript.done","text":"missing duration"}));
        peer.finish();
        let (driver_result, events) = join(driver, drain_events(events)).await;
        assert_eq!(driver_result, Err(RealtimeError::UnexpectedRemoteClose));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, XaiSttEvent::ProviderEvent { name, .. } if name == "transcript.done"))
                .count(),
            2
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, XaiSttEvent::Completed { .. })));
    });
}

#[test]
fn transport_error_after_valid_done_is_never_treated_as_normal_eof() {
    block_on(async {
        let (fake, peer) = fake_transport(&[ready()]);
        let (session, driver) =
            XaiSttSession::connect(fake, credential(), XaiSttConfig::default(), limits())
                .await
                .unwrap();
        let (control, events) = session.into_parts();
        control.finish_audio().unwrap();
        let mut driver = Box::pin(driver.run());
        progress_until_sent(&mut driver, &peer.sent, 1).await;
        peer.push(json!({"type":"transcript.done","text":"valid","duration":0.5}));
        peer.fail(RealtimeError::Transport {
            message: "socket read failed".into(),
        });
        let (driver_result, events) = join(driver, drain_events(events)).await;
        assert_eq!(
            driver_result,
            Err(RealtimeError::Transport {
                message: "socket read failed".into()
            })
        );
        assert!(events.iter().any(|event| matches!(event,
            XaiSttEvent::ConnectionInterrupted { message } if message == "realtime transport failed: socket read failed")));
        assert!(!events
            .iter()
            .any(|event| matches!(event, XaiSttEvent::Completed { .. })));
    });
}

#[test]
fn queued_audio_done_is_not_completion_when_socket_send_fails() {
    block_on(async {
        let (fake, peer) = fake_transport_with_send_error(
            &[ready()],
            Some(RealtimeError::Transport {
                message: "socket write failed".into(),
            }),
        );
        let (session, driver) =
            XaiSttSession::connect(fake, credential(), XaiSttConfig::default(), limits())
                .await
                .unwrap();
        let (control, events) = session.into_parts();
        control.finish_audio().unwrap();
        let (driver_result, events) = join(driver.run(), drain_events(events)).await;
        assert_eq!(
            driver_result,
            Err(RealtimeError::Transport {
                message: "socket write failed".into()
            })
        );
        assert!(peer.sent.lock().unwrap().is_empty());
        assert!(!events
            .iter()
            .any(|event| matches!(event, XaiSttEvent::Completed { .. })));
    });
}

#[test]
fn ready_wait_timeout_and_invalid_config_happen_before_transport_use() {
    block_on(async {
        let (fake, peer) = fake_transport(&[]);
        let config = XaiSttConfig {
            connect_timeout: Duration::from_millis(5),
            ..XaiSttConfig::default()
        };
        let result = XaiSttSession::connect(fake, credential(), config, limits()).await;
        assert!(matches!(
            result,
            Err(
                lingxi_llm_client::providers::xai::stt::XaiSttError::Realtime(
                    RealtimeError::Transport { .. }
                )
            )
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);

        let (fake, peer) = fake_transport(&[]);
        let invalid = XaiSttConfig {
            encoding: XaiSttEncoding::Opus,
            sample_rate_hz: Some(16_000),
            ..XaiSttConfig::default()
        };
        let result = XaiSttSession::connect(fake, credential(), invalid, limits()).await;
        assert!(matches!(
            result,
            Err(
                lingxi_llm_client::providers::xai::stt::XaiSttError::Realtime(
                    RealtimeError::InvalidConfig { .. }
                )
            )
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
    });
}

impl FakePeer {
    fn push(&self, value: Value) {
        self.incoming
            .as_ref()
            .expect("peer stream is open")
            .unbounded_send(Ok(RealtimeFrame::text(value.to_string())))
            .unwrap();
    }

    fn fail(&self, error: RealtimeError) {
        self.incoming
            .as_ref()
            .expect("peer stream is open")
            .unbounded_send(Err(error))
            .unwrap();
    }

    fn finish(&mut self) {
        self.incoming.take();
    }
}
