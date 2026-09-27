use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::{mpsc, oneshot},
    executor::block_on,
    stream, StreamExt,
};
use lingxi_llm_client::{
    files::provider_file_endpoint_fingerprint,
    minimax_bidi_tts::{
        MiniMaxBidiTtsConfig, MiniMaxBidiTtsCredentials, MiniMaxBidiTtsError,
        MiniMaxBidiTtsEventKind, MiniMaxBidiTtsLimits, MiniMaxBidiTtsRegion, MiniMaxBidiTtsRequest,
        MiniMaxBidiTtsService, MINIMAX_BIDI_TTS_CHINA_ENDPOINT,
        MINIMAX_BIDI_TTS_INTERNATIONAL_ENDPOINT,
    },
    minimax_voices::{MiniMaxVoiceKind, MiniMaxVoiceRef, MiniMaxVoicesRegion, MiniMaxVoicesScope},
    protocol::{ProviderId, Secret},
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeSink, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::{sync::Arc, sync::Mutex, time::Duration};

type Incoming = Result<RealtimeFrame, RealtimeError>;

#[derive(Clone, Copy)]
enum StartBehavior {
    Ack,
    NoAck,
}

struct Shared {
    outbound: Mutex<Vec<Value>>,
    connect_request: Mutex<Option<RealtimeConnectRequest>>,
    connect_count: Mutex<usize>,
    close_count: Mutex<usize>,
    inbound_sender: Mutex<Option<mpsc::UnboundedSender<Incoming>>>,
    send_pending: Mutex<bool>,
    send_release: Mutex<Option<oneshot::Sender<()>>>,
}

struct FakeTransport {
    shared: Arc<Shared>,
    initial: Mutex<Option<Vec<Incoming>>>,
    start_behavior: StartBehavior,
    hold_input_sends: bool,
    auto_finish: bool,
    send_release: Mutex<Option<oneshot::Receiver<()>>>,
}

struct FakePeer {
    shared: Arc<Shared>,
}

struct FakeSink {
    shared: Arc<Shared>,
    start_behavior: StartBehavior,
    hold_input_sends: bool,
    auto_finish: bool,
    send_release: Option<oneshot::Receiver<()>>,
}

fn fake_transport() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with(
        vec![Ok(RealtimeFrame::text(connected_success().to_string()))],
        StartBehavior::Ack,
    )
}

fn fake_transport_with(
    initial: Vec<Incoming>,
    start_behavior: StartBehavior,
) -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_options(initial, start_behavior, false, true)
}

fn fake_transport_holding_input_sends() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_options(
        vec![Ok(RealtimeFrame::text(connected_success().to_string()))],
        StartBehavior::Ack,
        true,
        true,
    )
}

fn fake_transport_manual_finish() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_options(
        vec![Ok(RealtimeFrame::text(connected_success().to_string()))],
        StartBehavior::Ack,
        false,
        false,
    )
}

fn fake_transport_with_options(
    initial: Vec<Incoming>,
    start_behavior: StartBehavior,
    hold_input_sends: bool,
    auto_finish: bool,
) -> (Arc<FakeTransport>, FakePeer) {
    let (send_release, send_release_rx) = oneshot::channel();
    let shared = Arc::new(Shared {
        outbound: Mutex::new(Vec::new()),
        connect_request: Mutex::new(None),
        connect_count: Mutex::new(0),
        close_count: Mutex::new(0),
        inbound_sender: Mutex::new(None),
        send_pending: Mutex::new(false),
        send_release: Mutex::new(hold_input_sends.then_some(send_release)),
    });
    (
        Arc::new(FakeTransport {
            shared: shared.clone(),
            initial: Mutex::new(Some(initial)),
            start_behavior,
            hold_input_sends,
            auto_finish,
            send_release: Mutex::new(hold_input_sends.then_some(send_release_rx)),
        }),
        FakePeer { shared },
    )
}

#[async_trait]
impl RealtimeTransport for FakeTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        *self.shared.connect_request.lock().unwrap() = Some(request);
        *self.shared.connect_count.lock().unwrap() += 1;

        let (sender, receiver) = mpsc::unbounded();
        *self.shared.inbound_sender.lock().unwrap() = Some(sender);
        let initial = self
            .initial
            .lock()
            .unwrap()
            .take()
            .expect("fake transport connects once");
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                shared: self.shared.clone(),
                start_behavior: self.start_behavior,
                hold_input_sends: self.hold_input_sends,
                auto_finish: self.auto_finish,
                send_release: self.send_release.lock().unwrap().take(),
            }),
            inbound: stream::iter(initial).chain(receiver).boxed(),
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        let RealtimeFrame::Text(bytes) = frame else {
            return Err(RealtimeError::InvalidInput {
                message: "MiniMax bidi TTS sends JSON text frames".into(),
            });
        };
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|error| RealtimeError::InvalidInput {
                message: format!("outbound JSON was invalid: {error}"),
            })?;
        self.shared.outbound.lock().unwrap().push(value.clone());
        if self.hold_input_sends
            && matches!(
                value.get("event").and_then(Value::as_str),
                Some("task_continue" | "task_flush" | "task_cancel")
            )
        {
            *self.shared.send_pending.lock().unwrap() = true;
            if let Some(receiver) = self.send_release.take() {
                let _ = receiver.await;
            }
            *self.shared.send_pending.lock().unwrap() = false;
        }
        match value.get("event").and_then(Value::as_str) {
            Some("task_start") if matches!(self.start_behavior, StartBehavior::Ack) => {
                self.respond(task_event("task_started"));
            }
            Some("task_finish") if self.auto_finish => {
                self.respond(task_event("task_finished"));
                self.shared.inbound_sender.lock().unwrap().take();
            }
            _ => {}
        }
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        *self.shared.close_count.lock().unwrap() += 1;
        self.shared.inbound_sender.lock().unwrap().take();
        Ok(())
    }
}

impl FakeSink {
    fn respond(&self, event: Value) {
        let sender = self.shared.inbound_sender.lock().unwrap().clone();
        if let Some(sender) = sender {
            let _ = sender.unbounded_send(Ok(RealtimeFrame::text(event.to_string())));
        }
    }
}

impl FakePeer {
    fn respond(&self, event: Value) {
        let sender = self.shared.inbound_sender.lock().unwrap().clone();
        if let Some(sender) = sender {
            let _ = sender.unbounded_send(Ok(RealtimeFrame::text(event.to_string())));
        }
    }

    fn respond_raw(&self, frame: RealtimeFrame) {
        let sender = self.shared.inbound_sender.lock().unwrap().clone();
        if let Some(sender) = sender {
            let _ = sender.unbounded_send(Ok(frame));
        }
    }

    fn fail_inbound(&self, message: &str) {
        let sender = self.shared.inbound_sender.lock().unwrap().clone();
        if let Some(sender) = sender {
            let _ = sender.unbounded_send(Err(RealtimeError::Transport {
                message: message.to_owned(),
            }));
        }
    }

    fn close_inbound(&self) {
        self.shared.inbound_sender.lock().unwrap().take();
    }

    fn outbound(&self) -> Vec<Value> {
        self.shared.outbound.lock().unwrap().clone()
    }

    fn connect_request(&self) -> RealtimeConnectRequest {
        self.shared
            .connect_request
            .lock()
            .unwrap()
            .clone()
            .expect("transport request captured")
    }

    fn connect_count(&self) -> usize {
        *self.shared.connect_count.lock().unwrap()
    }

    fn close_count(&self) -> usize {
        *self.shared.close_count.lock().unwrap()
    }

    fn send_pending(&self) -> bool {
        *self.shared.send_pending.lock().unwrap()
    }

    fn release_input_send(&self) {
        if let Some(sender) = self.shared.send_release.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }
}

fn connected_success() -> Value {
    task_event("connected_success")
}

fn task_event(event: &str) -> Value {
    json!({
        "session_id": "bidi-session-1",
        "connect_id": "bidi-connect-1",
        "event": event,
        "trace_id": format!("trace-{event}"),
        "base_resp": { "status_code": 0, "status_msg": "success" }
    })
}

fn sentence_event(event: &str) -> Value {
    task_event(event)
}

fn audio_event(hex: &str, is_final: bool) -> Value {
    // The documented Bidi audio response has no event field.
    json!({
        "data": { "audio": hex },
        "is_final": is_final,
        "session_id": "bidi-session-1",
        "connect_id": "bidi-connect-1",
        "trace_id": "trace-audio",
        "base_resp": { "status_code": 0, "status_msg": "success" }
    })
}

fn make_service(
    transport: Arc<FakeTransport>,
    region: MiniMaxBidiTtsRegion,
) -> Result<MiniMaxBidiTtsService, MiniMaxBidiTtsError> {
    MiniMaxBidiTtsService::new(
        transport,
        MiniMaxBidiTtsConfig::new("bidi-profile", "team/bidi-account", region),
    )
}

fn credentials() -> MiniMaxBidiTtsCredentials {
    MiniMaxBidiTtsCredentials::new(Secret::new("bidi-test-key".to_owned()))
}

fn request() -> MiniMaxBidiTtsRequest {
    MiniMaxBidiTtsRequest::new("English_expressive_narrator").with_session_id("bidi-session-1")
}

fn voice_reference(
    profile_name: &str,
    account_scope: &str,
    endpoint_root: &str,
    region: MiniMaxVoicesRegion,
) -> MiniMaxVoiceRef {
    MiniMaxVoiceRef {
        scope: MiniMaxVoicesScope {
            provider_id: ProviderId::new("minimax"),
            profile_name: profile_name.to_owned(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(endpoint_root),
            account_scope: account_scope.to_owned(),
            region,
        },
        kind: MiniMaxVoiceKind::Cloned,
        voice_id: "bidi-created-voice".into(),
    }
}

#[test]
fn regional_routes_auth_and_task_start_are_exact() {
    block_on(async {
        for (region, endpoint) in [
            (
                MiniMaxBidiTtsRegion::International,
                MINIMAX_BIDI_TTS_INTERNATIONAL_ENDPOINT,
            ),
            (
                MiniMaxBidiTtsRegion::ChinaMainland,
                MINIMAX_BIDI_TTS_CHINA_ENDPOINT,
            ),
        ] {
            let (fake, peer) = fake_transport();
            let active_service = make_service(fake, region).unwrap();
            let _session = active_service
                .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
                .await
                .unwrap();

            let connect = peer.connect_request();
            assert_eq!(connect.endpoint, endpoint);
            assert_eq!(connect.headers.len(), 1);
            assert_eq!(connect.headers[0].0, "Authorization");
            assert_eq!(connect.headers[0].1, "Bearer bidi-test-key");
            assert_eq!(peer.connect_count(), 1);

            let sent = peer.outbound();
            assert_eq!(sent.len(), 1);
            assert_eq!(sent[0]["event"], "task_start");
            assert_eq!(sent[0]["session_id"], "bidi-session-1");
            assert_eq!(
                sent[0]["voice_setting"]["voice_id"],
                "English_expressive_narrator"
            );
            assert_eq!(sent[0]["model"], "speech-2.8-hd");
        }
    });
}

#[test]
fn missing_connected_success_fails_before_task_start() {
    block_on(async {
        let (fake, peer) = fake_transport_with(
            vec![Ok(RealtimeFrame::text(
                task_event("unexpected_ready_event").to_string(),
            ))],
            StartBehavior::Ack,
        );
        let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        assert!(service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .is_err());
        assert_eq!(peer.connect_count(), 1);
        assert!(peer.outbound().is_empty());
    });
}

#[test]
fn missing_task_started_ack_times_out_without_reconnecting() {
    block_on(async {
        let (fake, peer) = fake_transport_with(
            vec![Ok(RealtimeFrame::text(connected_success().to_string()))],
            StartBehavior::NoAck,
        );
        let config = MiniMaxBidiTtsConfig::new(
            "bidi-profile",
            "team/bidi-account",
            MiniMaxBidiTtsRegion::International,
        )
        .with_connect_timeout(Duration::from_millis(100));
        let service = MiniMaxBidiTtsService::new(fake, config).unwrap();
        assert!(service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .is_err());
        assert_eq!(peer.connect_count(), 1);
        assert_eq!(peer.outbound().len(), 1);
        assert_eq!(peer.outbound()[0]["event"], "task_start");
    });
}

#[test]
fn text_limit_counts_unicode_codepoints_and_rejects_the_next_character() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let limits = MiniMaxBidiTtsLimits {
            max_frame_bytes: 128 * 1024,
            ..MiniMaxBidiTtsLimits::default()
        };
        let session = service
            .connect(&credentials(), &request(), limits)
            .await
            .unwrap();
        let (mut input, _events) = session.into_parts();

        input.continue_text("  \n\t").await.unwrap();
        input.continue_text("").await.unwrap();
        let exact_limit = "界".repeat(10_000);
        input.continue_text(&exact_limit).await.unwrap();
        assert_eq!(
            peer.outbound()[3]["text"].as_str().unwrap().chars().count(),
            10_000
        );
        assert_eq!(peer.outbound()[1]["text"], "  \n\t");
        assert_eq!(peer.outbound()[2]["text"], "");

        let over_limit = "界".repeat(10_001);
        assert!(input.continue_text(&over_limit).await.is_err());
        assert_eq!(peer.outbound().len(), 4);
        assert_eq!(peer.connect_count(), 1);
    });
}

#[test]
fn sentence_and_audio_final_events_do_not_finish_the_session() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let session = service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();
        input.continue_text("First sentence.").await.unwrap();
        input.continue_text("Second sentence.").await.unwrap();

        peer.respond(sentence_event("sentence_start"));
        let official_audio_frame = audio_event("4142", true);
        assert!(official_audio_frame.get("event").is_none());
        peer.respond(official_audio_frame);
        peer.respond(sentence_event("sentence_end"));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::SentenceStart
        ));
        let audio = events.next().await.unwrap().unwrap();
        assert!(matches!(
            audio.kind,
            MiniMaxBidiTtsEventKind::AudioDelta {
                ref audio,
                is_final: true,
                ..
            } if audio == &Bytes::from_static(b"AB")
        ));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::SentenceEnd
        ));

        input
            .continue_text("The same session remains open.")
            .await
            .unwrap();
        assert_eq!(peer.connect_count(), 1);
        assert_eq!(peer.outbound().len(), 4);
    });
}

#[tokio::test]
async fn server_soft_errors_are_preserved_without_automatic_resend() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
    let session = service
        .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();

    for (index, code) in [2204, 2205].into_iter().enumerate() {
        let text = format!("caller-owned retry attempt {index}");
        input.continue_text(&text).await.unwrap();
        let mut error = task_event("task_failed");
        error["base_resp"]["status_code"] = json!(code);
        error["base_resp"]["status_msg"] = json!(format!("soft error {code}"));
        peer.respond(error.clone());

        let received = events.next().await.unwrap().unwrap();
        assert!(matches!(
            received.kind,
            MiniMaxBidiTtsEventKind::SoftError {
                status_code: actual,
                ..
            } if actual == code
        ));
        assert_eq!(received.native, Some(error));
        assert_eq!(
            peer.outbound()
                .iter()
                .filter(|value| value["event"] == "task_continue")
                .count(),
            index + 1
        );
        assert_eq!(peer.connect_count(), 1);
    }

    input
        .continue_text("provider failure is terminal")
        .await
        .unwrap();
    let mut fatal = task_event("task_failed");
    fatal["base_resp"]["status_code"] = json!(1004);
    fatal["base_resp"]["status_msg"] = json!("fatal provider failure");
    peer.respond(fatal);
    let received = events.next().await.unwrap().unwrap();
    assert!(matches!(
        received.kind,
        MiniMaxBidiTtsEventKind::TaskFailed {
            status_code: Some(1004),
            ..
        }
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(200), events.next())
            .await
            .expect("fatal task failure should close the event stream")
            .unwrap()
            .is_none()
    );
    assert_eq!(peer.close_count(), 1);
    assert!(matches!(
        input
            .continue_text("must not be sent after fatal failure")
            .await,
        Err(MiniMaxBidiTtsError::Closed)
    ));
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|value| value["event"] == "task_continue")
            .count(),
        3
    );
    assert_eq!(peer.connect_count(), 1);
}

#[test]
fn flush_is_ordered_after_audio_and_gates_new_text_until_ack() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let session = service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();
        input.continue_text("Flush this turn.").await.unwrap();
        input.flush().await.unwrap();
        assert!(input.continue_text("wait for flush ack").await.is_err());

        peer.respond(sentence_event("sentence_start"));
        peer.respond(audio_event("4142", true));
        peer.respond(sentence_event("sentence_end"));
        peer.respond(task_event("task_flushed"));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::SentenceStart
        ));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::AudioDelta { .. }
        ));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::SentenceEnd
        ));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::TaskFlushed
        ));
        input.continue_text("Continue after flush.").await.unwrap();
        assert_eq!(
            peer.outbound()
                .iter()
                .filter(|value| value["event"] == "task_flush")
                .count(),
            1
        );
        assert_eq!(peer.connect_count(), 1);
    });
}

#[test]
fn cancel_ack_releases_the_session_for_barge_in_text() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let session = service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();
        input.continue_text("Interruptible turn.").await.unwrap();
        input.flush().await.unwrap();
        input.cancel().await.unwrap();
        assert!(input.continue_text("wait for cancel ack").await.is_err());

        peer.respond(task_event("task_flushed"));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::TaskFlushed
        ));
        assert!(input
            .continue_text("late flush cannot unlock cancel")
            .await
            .is_err());

        peer.respond(task_event("task_canceled"));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::TaskCanceled
        ));
        input
            .continue_text("Barge-in continues here.")
            .await
            .unwrap();
        assert_eq!(
            peer.outbound()
                .iter()
                .filter(|value| value["event"] == "task_cancel")
                .count(),
            1
        );
        assert_eq!(
            peer.outbound()
                .iter()
                .filter(|value| value["event"] == "task_flush")
                .count(),
            1
        );
        assert_eq!(peer.connect_count(), 1);
    });
}

#[tokio::test]
async fn inbound_audio_is_processed_while_outbound_send_is_pending() {
    let (fake, peer) = fake_transport_holding_input_sends();
    let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
    let session = service
        .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();

    let sending = input.continue_text("A deliberately pending send.");
    tokio::pin!(sending);
    assert!(futures::poll!(&mut sending).is_pending());
    assert!(peer.send_pending());

    peer.respond(audio_event("4142", false));
    let event = tokio::time::timeout(Duration::from_millis(200), events.next())
        .await
        .expect("inbound event should progress during a pending send")
        .unwrap()
        .unwrap();
    assert!(matches!(
        event.kind,
        MiniMaxBidiTtsEventKind::AudioDelta {
            ref audio,
            is_final: false,
            ..
        } if audio == &Bytes::from_static(b"AB")
    ));

    peer.release_input_send();
    tokio::time::timeout(Duration::from_millis(200), sending)
        .await
        .expect("outbound send should complete after release")
        .unwrap();
    assert!(!peer.send_pending());
}

#[tokio::test]
async fn control_ack_arriving_before_send_returns_is_not_lost() {
    let (fake, peer) = fake_transport_holding_input_sends();
    let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
    let session = service
        .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();

    {
        let flushing = input.flush();
        tokio::pin!(flushing);
        assert!(futures::poll!(&mut flushing).is_pending());
        assert!(peer.send_pending());
        peer.respond(task_event("task_flushed"));
        assert!(matches!(
            tokio::time::timeout(Duration::from_millis(200), events.next())
                .await
                .expect("control event should progress during a pending send")
                .unwrap()
                .unwrap()
                .kind,
            MiniMaxBidiTtsEventKind::TaskFlushed
        ));

        peer.release_input_send();
        tokio::time::timeout(Duration::from_millis(200), flushing)
            .await
            .expect("flush send should complete after release")
            .unwrap();
    }
    input
        .continue_text("The early ack restored input readiness.")
        .await
        .unwrap();
    assert_eq!(peer.connect_count(), 1);
}

#[test]
fn finish_is_sent_once_and_clean_eof_requires_task_finished() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let first_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let session = first_service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();
        input
            .continue_text("Drain the final sentence.")
            .await
            .unwrap();
        input.finish().await.unwrap();

        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::TaskFinished
        ));
        assert!(events.next().await.unwrap().is_none());
        let sent = peer.outbound();
        assert_eq!(
            sent.iter()
                .filter(|value| value["event"] == "task_finish")
                .count(),
            1
        );
    });
}

#[test]
fn premature_eof_and_late_transport_error_are_not_clean_completion() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let premature_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let session = premature_service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();
        peer.close_inbound();
        assert!(events.next().await.is_err());
        assert!(matches!(
            input.continue_text("must not be sent after EOF").await,
            Err(MiniMaxBidiTtsError::Closed)
        ));
        assert_eq!(
            peer.outbound()
                .iter()
                .filter(|value| value["event"] == "task_continue")
                .count(),
            0
        );
        input.abort().await.unwrap();
        assert_eq!(peer.close_count(), 1);
        assert_eq!(peer.connect_count(), 1);

        let (fake, peer) = fake_transport_manual_finish();
        let late_error_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let session = late_error_service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();
        input.finish().await.unwrap();
        peer.respond(task_event("task_finished"));
        peer.fail_inbound("late frame read failure");
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            MiniMaxBidiTtsEventKind::TaskFinished
        ));
        assert!(events.next().await.is_err());
        assert_eq!(peer.connect_count(), 1);
    });
}

#[tokio::test]
async fn dropping_pending_input_and_control_sends_closes_the_session() {
    // A continue whose send outcome is unknown must not be retried after the
    // caller drops the in-flight future.
    let (fake, peer) = fake_transport_holding_input_sends();
    let continue_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
    let session = continue_service
        .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    let reading = events.next();
    tokio::pin!(reading);
    assert!(futures::poll!(&mut reading).is_pending());
    {
        let sending = input.continue_text("the send may already have reached MiniMax");
        tokio::pin!(sending);
        assert!(futures::poll!(&mut sending).is_pending());
        assert!(peer.send_pending());
    }
    let read_result = tokio::time::timeout(Duration::from_millis(200), reading)
        .await
        .expect("dropping a pending send should wake the blocked event reader");
    assert!(
        matches!(read_result, Ok(None) | Err(_)),
        "the blocked event reader must receive a terminal result"
    );
    assert!(matches!(
        input.continue_text("must not retry the unknown send").await,
        Err(MiniMaxBidiTtsError::Closed)
    ));
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|value| value["event"] == "task_continue")
            .count(),
        1
    );
    input.abort().await.unwrap();
    assert_eq!(peer.close_count(), 1);

    // Even an early flush acknowledgement cannot restore readiness after the
    // caller abandons the still-pending outbound send.
    let (fake, peer) = fake_transport_holding_input_sends();
    let flush_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
    let session = flush_service
        .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    {
        let flushing = input.flush();
        tokio::pin!(flushing);
        assert!(futures::poll!(&mut flushing).is_pending());
        assert!(peer.send_pending());
        peer.respond(task_event("task_flushed"));
        assert!(matches!(
            tokio::time::timeout(Duration::from_millis(200), events.next())
                .await
                .expect("early flush acknowledgement should be readable")
                .unwrap()
                .unwrap()
                .kind,
            MiniMaxBidiTtsEventKind::TaskFlushed
        ));
    }
    assert!(matches!(
        input
            .continue_text("early ack cannot confirm a dropped send")
            .await,
        Err(MiniMaxBidiTtsError::Closed)
    ));
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|value| value["event"] == "task_flush")
            .count(),
        1
    );
    input.abort().await.unwrap();
    assert_eq!(peer.close_count(), 1);

    // A dropped cancel send has the same ambiguous outcome and must not leave
    // the task permanently gated in a retryable pending state.
    let (fake, peer) = fake_transport_holding_input_sends();
    let cancel_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
    let session = cancel_service
        .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
        .await
        .unwrap();
    let (mut input, _events) = session.into_parts();
    {
        let canceling = input.cancel();
        tokio::pin!(canceling);
        assert!(futures::poll!(&mut canceling).is_pending());
        assert!(peer.send_pending());
    }
    assert!(matches!(
        input
            .continue_text("must not send after dropped cancel")
            .await,
        Err(MiniMaxBidiTtsError::Closed)
    ));
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|value| value["event"] == "task_cancel")
            .count(),
        1
    );
    input.abort().await.unwrap();
    assert_eq!(peer.close_count(), 1);
}

#[test]
fn unknown_and_malformed_events_keep_their_original_payloads() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let session = service
            .connect(&credentials(), &request(), MiniMaxBidiTtsLimits::default())
            .await
            .unwrap();
        let (_input, mut events) = session.into_parts();
        let unknown = json!({
            "event": "future_bidi_event",
            "session_id": "bidi-session-1",
            "vendor_field": { "version": 7 },
            "data": { "audio": "4142" },
            "is_final": true,
            "base_resp": { "status_code": 0, "status_msg": "success" }
        });
        peer.respond(unknown.clone());
        let malformed = json!({
            "event": "task_continued",
            "session_id": "bidi-session-1",
            "connect_id": "bidi-connect-1",
            "data": { "audio": "not-hex" },
            "is_final": false,
            "base_resp": { "status_code": 0, "status_msg": "success" },
            "extension": "preserve"
        });
        peer.respond_raw(RealtimeFrame::text(malformed.to_string()));
        peer.respond_raw(RealtimeFrame::text("{broken-json"));

        let first = events.next().await.unwrap().unwrap();
        assert!(matches!(
            first.kind,
            MiniMaxBidiTtsEventKind::Native { ref event }
                if event.as_deref() == Some("future_bidi_event")
        ));
        assert_eq!(first.native, Some(unknown));

        let second = events.next().await.unwrap().unwrap();
        assert!(matches!(
            second.kind,
            MiniMaxBidiTtsEventKind::Malformed { .. }
        ));
        assert_eq!(second.native, Some(malformed));

        let third = events.next().await.unwrap().unwrap();
        assert!(matches!(
            third.kind,
            MiniMaxBidiTtsEventKind::Malformed { .. }
        ));
        assert_eq!(third.native, None);
        assert_eq!(third.raw_frame, Some(Bytes::from_static(b"{broken-json")));
    });
}

#[test]
fn frame_and_ping_bounds_fail_locally_and_unsupported_ping_keeps_session_open() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let first_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let limits = MiniMaxBidiTtsLimits {
            max_frame_bytes: 1024,
            ..MiniMaxBidiTtsLimits::default()
        };
        let session = first_service
            .connect(&credentials(), &request(), limits)
            .await
            .unwrap();
        let (mut input, _events) = session.into_parts();
        let text_exceeds_frame = "界".repeat(400);
        assert!(input.continue_text(&text_exceeds_frame).await.is_err());
        assert_eq!(peer.outbound().len(), 1);

        assert!(input.ping(Bytes::from(vec![0_u8; 126])).await.is_err());
        assert!(input.ping(Bytes::from_static(b"keepalive")).await.is_err());
        input
            .continue_text("still usable after unsupported ping")
            .await
            .unwrap();
        assert_eq!(peer.outbound().len(), 2);
        assert_eq!(
            peer.outbound()[1]["text"],
            "still usable after unsupported ping"
        );

        let (fake, peer) = fake_transport();
        let second_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let audio_limits = MiniMaxBidiTtsLimits {
            max_audio_bytes_per_session: 2,
            ..MiniMaxBidiTtsLimits::default()
        };
        let session = second_service
            .connect(&credentials(), &request(), audio_limits)
            .await
            .unwrap();
        let (_input, mut events) = session.into_parts();
        peer.respond(audio_event("414243", false));
        assert!(events.next().await.is_err());
    });
}

#[test]
fn scoped_voice_mismatch_fails_before_connect_and_matching_ref_sets_wire_id() {
    block_on(async {
        let (fake, peer) = fake_transport();
        let active_service = make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
        let valid_voice = MiniMaxVoiceRef::new(
            voice_reference(
                "bidi-profile",
                "team/bidi-account",
                "https://api.minimax.io/v1",
                MiniMaxVoicesRegion::International,
            )
            .scope,
            MiniMaxVoiceKind::Cloned,
            "bidi-created-voice",
        )
        .unwrap();
        let _session = active_service
            .connect_with_voice(
                &credentials(),
                &request(),
                &valid_voice,
                MiniMaxBidiTtsLimits::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            peer.outbound()[0]["voice_setting"]["voice_id"],
            "bidi-created-voice"
        );

        for voice in [
            voice_reference(
                "another-profile",
                "team/bidi-account",
                "https://api.minimax.io/v1",
                MiniMaxVoicesRegion::International,
            ),
            voice_reference(
                "bidi-profile",
                "another-account",
                "https://api.minimax.io/v1",
                MiniMaxVoicesRegion::International,
            ),
            voice_reference(
                "bidi-profile",
                "team/bidi-account",
                "https://api.minimax.io:444/v1",
                MiniMaxVoicesRegion::International,
            ),
        ] {
            let (fake, peer) = fake_transport();
            let mismatched_service =
                make_service(fake, MiniMaxBidiTtsRegion::International).unwrap();
            assert!(mismatched_service
                .connect_with_voice(
                    &credentials(),
                    &request(),
                    &voice,
                    MiniMaxBidiTtsLimits::default(),
                )
                .await
                .is_err());
            assert_eq!(peer.connect_count(), 0);
            assert!(peer.outbound().is_empty());
        }
    });
}
