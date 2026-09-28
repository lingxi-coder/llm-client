use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::{mpsc, oneshot},
    stream::BoxStream,
    Stream, StreamExt,
};
use lingxi_llm_client::{
    client::RequestOptions,
    protocol::Secret,
    providers::qwen::tts::QwenTtsLanguage,
    providers::qwen::tts_realtime::{
        QwenTtsRealtimeConfig, QwenTtsRealtimeEventKind, QwenTtsRealtimeFormat,
        QwenTtsRealtimeLimits, QwenTtsRealtimeMode, QwenTtsRealtimeRegion, QwenTtsRealtimeRequest,
        QwenTtsRealtimeService, QwenTtsRealtimeVoice,
    },
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeSink, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::{
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering},
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct Shared {
    outbound: Mutex<Vec<Value>>,
    connect_request: Mutex<Option<RealtimeConnectRequest>>,
    connect_count: Mutex<usize>,
    close_count: Mutex<usize>,
    close_pending: Mutex<bool>,
    finish_send_pending: Mutex<bool>,
    inbound_sender: Mutex<Option<mpsc::UnboundedSender<Incoming>>>,
    send_pending: Mutex<bool>,
    _send_release: Mutex<Option<oneshot::Sender<()>>>,
    _close_release: Mutex<Option<oneshot::Sender<()>>>,
    _finish_send_release: Mutex<Option<oneshot::Sender<()>>>,
    updated_language_override: Mutex<Option<String>>,
    omit_updated_language: Mutex<bool>,
    eof_poll_count: Arc<AtomicUsize>,
}

struct FakeTransport {
    shared: Arc<Shared>,
    initial: Mutex<Option<Vec<Incoming>>>,
    auto_update_ack: bool,
    auto_finish: bool,
    hold_appends: bool,
    hold_close: bool,
    hold_finish_send: bool,
    send_release: Mutex<Option<oneshot::Receiver<()>>>,
    close_release: Mutex<Option<oneshot::Receiver<()>>>,
    finish_send_release: Mutex<Option<oneshot::Receiver<()>>>,
}

struct FakePeer {
    shared: Arc<Shared>,
}

struct FakeSink {
    shared: Arc<Shared>,
    auto_update_ack: bool,
    auto_finish: bool,
    hold_appends: bool,
    hold_close: bool,
    hold_finish_send: bool,
    send_release: Option<oneshot::Receiver<()>>,
    close_release: Option<oneshot::Receiver<()>>,
    finish_send_release: Option<oneshot::Receiver<()>>,
}

struct EofCountingStream {
    inner: BoxStream<'static, Incoming>,
    eof_poll_count: Arc<AtomicUsize>,
    ended: bool,
}

impl Stream for EofCountingStream {
    type Item = Incoming;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        assert!(!this.ended, "fake inbound stream polled after EOF");
        match this.inner.as_mut().poll_next(context) {
            Poll::Ready(None) => {
                this.ended = true;
                this.eof_poll_count.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(None)
            }
            result => result,
        }
    }
}

fn fake_transport() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with(Vec::new(), true, true, false)
}

fn fake_transport_manual_finish() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with(Vec::new(), true, false, false)
}

fn fake_transport_holding_appends() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with(Vec::new(), true, true, true)
}

fn fake_transport_with_language_mismatch() -> (Arc<FakeTransport>, FakePeer) {
    let (transport, peer) = fake_transport();
    *transport.shared.updated_language_override.lock().unwrap() = Some("Chinese".into());
    (transport, peer)
}

fn fake_transport_without_updated_language() -> (Arc<FakeTransport>, FakePeer) {
    let (transport, peer) = fake_transport();
    *transport.shared.omit_updated_language.lock().unwrap() = true;
    (transport, peer)
}

fn fake_transport_with(
    extra_initial: Vec<Incoming>,
    auto_update_ack: bool,
    auto_finish: bool,
    hold_appends: bool,
) -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_gates(
        extra_initial,
        auto_update_ack,
        auto_finish,
        hold_appends,
        false,
        false,
    )
}

fn fake_transport_holding_close() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_gates(Vec::new(), true, false, false, true, false)
}

fn fake_transport_holding_finish_send() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_gates(Vec::new(), true, false, false, false, true)
}

fn fake_transport_with_gates(
    extra_initial: Vec<Incoming>,
    auto_update_ack: bool,
    auto_finish: bool,
    hold_appends: bool,
    hold_close: bool,
    hold_finish_send: bool,
) -> (Arc<FakeTransport>, FakePeer) {
    let (send_release, send_release_rx) = oneshot::channel();
    let (close_release, close_release_rx) = oneshot::channel();
    let (finish_send_release, finish_send_release_rx) = oneshot::channel();
    let shared = Arc::new(Shared {
        outbound: Mutex::new(Vec::new()),
        connect_request: Mutex::new(None),
        connect_count: Mutex::new(0),
        close_count: Mutex::new(0),
        close_pending: Mutex::new(false),
        finish_send_pending: Mutex::new(false),
        inbound_sender: Mutex::new(None),
        send_pending: Mutex::new(false),
        _send_release: Mutex::new(hold_appends.then_some(send_release)),
        _close_release: Mutex::new(hold_close.then_some(close_release)),
        _finish_send_release: Mutex::new(hold_finish_send.then_some(finish_send_release)),
        updated_language_override: Mutex::new(None),
        omit_updated_language: Mutex::new(false),
        eof_poll_count: Arc::new(AtomicUsize::new(0)),
    });
    let initial = std::iter::once(Ok(RealtimeFrame::text(session_created().to_string())))
        .chain(extra_initial)
        .collect();

    (
        Arc::new(FakeTransport {
            shared: shared.clone(),
            initial: Mutex::new(Some(initial)),
            auto_update_ack,
            auto_finish,
            hold_appends,
            hold_close,
            hold_finish_send,
            send_release: Mutex::new(hold_appends.then_some(send_release_rx)),
            close_release: Mutex::new(hold_close.then_some(close_release_rx)),
            finish_send_release: Mutex::new(hold_finish_send.then_some(finish_send_release_rx)),
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
        let requested_model = request
            .endpoint
            .rsplit_once("?model=")
            .map(|(_, model)| model.to_owned())
            .unwrap_or_else(|| "qwen3-tts-flash-realtime".into());
        *self.shared.connect_request.lock().unwrap() = Some(request);
        *self.shared.connect_count.lock().unwrap() += 1;
        let (sender, receiver) = mpsc::unbounded();
        *self.shared.inbound_sender.lock().unwrap() = Some(sender);
        let mut initial = self
            .initial
            .lock()
            .unwrap()
            .take()
            .expect("one connection per fake transport");
        if let Some(Ok(RealtimeFrame::Text(bytes))) = initial.first_mut() {
            let mut created: Value = serde_json::from_slice(bytes).unwrap();
            created["session"]["model"] = json!(requested_model);
            *bytes = Bytes::from(created.to_string());
        }
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                shared: self.shared.clone(),
                auto_update_ack: self.auto_update_ack,
                auto_finish: self.auto_finish,
                hold_appends: self.hold_appends,
                hold_close: self.hold_close,
                hold_finish_send: self.hold_finish_send,
                send_release: self.send_release.lock().unwrap().take(),
                close_release: self.close_release.lock().unwrap().take(),
                finish_send_release: self.finish_send_release.lock().unwrap().take(),
            }),
            inbound: EofCountingStream {
                inner: futures::stream::iter(initial).chain(receiver).boxed(),
                eof_poll_count: self.shared.eof_poll_count.clone(),
                ended: false,
            }
            .boxed(),
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        let RealtimeFrame::Text(bytes) = frame else {
            return Err(RealtimeError::InvalidInput {
                message: "Qwen TTS Realtime uses JSON text events".into(),
            });
        };
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|error| RealtimeError::InvalidInput {
                message: format!("outbound event was invalid JSON: {error}"),
            })?;
        self.shared.outbound.lock().unwrap().push(value.clone());

        if value.get("type").and_then(Value::as_str) == Some("session.finish")
            && self.hold_finish_send
        {
            *self.shared.finish_send_pending.lock().unwrap() = true;
            if let Some(receiver) = self.finish_send_release.take() {
                let _ = receiver.await;
            }
            *self.shared.finish_send_pending.lock().unwrap() = false;
        }

        match value.get("type").and_then(Value::as_str) {
            Some("session.update") if self.auto_update_ack => {
                let mut session = value.get("session").cloned().unwrap_or_else(|| json!({}));
                if !session.is_object() {
                    session = json!({});
                }
                session["id"] = json!("qwen-tts-session-1");
                session["object"] = json!("realtime.session");
                let model = self
                    .shared
                    .connect_request
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|request| request.endpoint.rsplit_once("?model="))
                    .map(|(_, model)| model.to_owned())
                    .unwrap_or_else(|| "qwen3-tts-flash-realtime".into());
                session["model"] = json!(model);
                if *self.shared.omit_updated_language.lock().unwrap() {
                    session.as_object_mut().unwrap().remove("language_type");
                } else if let Some(language) = self
                    .shared
                    .updated_language_override
                    .lock()
                    .unwrap()
                    .clone()
                {
                    session["language_type"] = json!(language);
                }
                self.respond(json!({
                    "event_id": "event-session-updated",
                    "type": "session.updated",
                    "session": session
                }));
            }
            Some("session.finish") if self.auto_finish => {
                self.respond(session_finished());
                self.shared.inbound_sender.lock().unwrap().take();
            }
            Some("input_text_buffer.commit") => {
                self.respond(json!({
                    "event_id": "event-buffer-committed",
                    "type": "input_text_buffer.committed",
                    "item_id": "user-item-1"
                }));
                self.respond(response_created("resp-commit"));
            }
            Some("input_text_buffer.clear") => {
                self.respond(json!({
                    "event_id": "event-buffer-cleared",
                    "type": "input_text_buffer.cleared"
                }));
            }
            Some("input_text_buffer.append") if self.hold_appends => {
                *self.shared.send_pending.lock().unwrap() = true;
                if let Some(receiver) = self.send_release.take() {
                    let _ = receiver.await;
                }
                *self.shared.send_pending.lock().unwrap() = false;
            }
            _ => {}
        }
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        *self.shared.close_count.lock().unwrap() += 1;
        if self.hold_close {
            *self.shared.close_pending.lock().unwrap() = true;
            if let Some(receiver) = self.close_release.take() {
                let _ = receiver.await;
            }
            *self.shared.close_pending.lock().unwrap() = false;
        }
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
            .expect("fake transport request captured")
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

    fn close_pending(&self) -> bool {
        *self.shared.close_pending.lock().unwrap()
    }

    fn release_close(&self) {
        if let Some(sender) = self.shared._close_release.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }

    fn finish_send_pending(&self) -> bool {
        *self.shared.finish_send_pending.lock().unwrap()
    }

    fn release_finish_send(&self) {
        if let Some(sender) = self.shared._finish_send_release.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }

    fn eof_poll_count(&self) -> usize {
        self.shared.eof_poll_count.load(Ordering::SeqCst)
    }
}

fn session_created() -> Value {
    json!({
        "event_id": "event-session-created",
        "type": "session.created",
        "session": {
            "object": "realtime.session",
            "mode": "server_commit",
            "model": "qwen3-tts-flash-realtime",
            "voice": "Cherry",
            "response_format": "pcm",
            "sample_rate": 24000,
            "id": "qwen-tts-session-1"
        }
    })
}

fn session_finished() -> Value {
    json!({
        "event_id": "event-session-finished",
        "type": "session.finished"
    })
}

fn make_service(
    transport: Arc<FakeTransport>,
    region: QwenTtsRealtimeRegion,
) -> Result<
    QwenTtsRealtimeService,
    lingxi_llm_client::providers::qwen::tts_realtime::QwenTtsRealtimeError,
> {
    QwenTtsRealtimeService::new(
        transport,
        QwenTtsRealtimeConfig::new(
            "qwen-tts-profile",
            "billing-account-7",
            region,
            "workspace-abc123",
        ),
    )
}

fn options(api_key: &str) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new(api_key.to_owned())),
        account_scope: Some("billing-account-7".into()),
        ..RequestOptions::default()
    }
}

async fn assert_preflight_rejected(region: QwenTtsRealtimeRegion, request: QwenTtsRealtimeRequest) {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, region).unwrap();
    assert!(service
        .connect(
            &options("regional-api-key"),
            &request,
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .is_err());
    assert_eq!(peer.connect_count(), 0);
    assert!(peer.outbound().is_empty());
}

fn make_request(mode: QwenTtsRealtimeMode) -> QwenTtsRealtimeRequest {
    let mut request = QwenTtsRealtimeRequest::new(
        "qwen3-tts-flash-realtime",
        QwenTtsRealtimeVoice::System("Cherry".into()),
    );
    request.mode = mode;
    request
}

fn configured_request(mode: QwenTtsRealtimeMode) -> QwenTtsRealtimeRequest {
    let mut request = make_request(mode);
    request.language_type = QwenTtsLanguage::English;
    request.format = QwenTtsRealtimeFormat::Opus;
    request.sample_rate = 48_000;
    request.speech_rate = Some(1.25);
    request.pitch_rate = Some(1.1);
    request.volume = Some(72);
    request.bit_rate = Some(160);
    request
}

fn sent_json(peer: &FakePeer) -> Vec<Value> {
    peer.outbound()
}

fn assert_client_event_ids_are_unique_uuids(events: &[Value]) {
    let ids = events
        .iter()
        .map(|event| {
            let id = event["event_id"]
                .as_str()
                .expect("each client event includes event_id");
            assert!(is_uuid(id), "client event_id must be a UUID: {id}");
            id
        })
        .collect::<Vec<_>>();
    for (index, id) in ids.iter().enumerate() {
        assert!(
            !ids[index + 1..].contains(id),
            "client event IDs must be unique within the WebSocket session"
        );
    }
}

fn is_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && bytes.iter().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                *byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

fn header<'a>(request: &'a RealtimeConnectRequest, name: &str) -> Option<&'a str> {
    request
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn audio_delta(response_id: &str, item_id: &str, delta: &str) -> Value {
    json!({
        "event_id": "event-audio-delta",
        "type": "response.audio.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta
    })
}

fn response_done(response_id: &str, status: &str, usage: Value) -> Value {
    json!({
        "event_id": "event-response-done",
        "type": "response.done",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "conversation_id": "",
            "status": status,
            "modalities": ["text", "audio"],
            "voice": "Cherry",
            "output": if status == "completed" {
                json!([{
                    "id": "item-81",
                    "object": "realtime.item",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{ "type": "audio", "transcript": "" }]
                }])
            } else {
                json!([])
            },
            "usage": usage
        }
    })
}

fn response_created(response_id: &str) -> Value {
    json!({
        "event_id": format!("event-created-{response_id}"),
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "modalities": ["text", "audio"],
            "voice": "Cherry",
            "output": []
        }
    })
}

#[tokio::test]
async fn beijing_and_singapore_use_official_routes_keys_workspace_and_model_query() {
    for (region, api_key, endpoint) in [
        (
            QwenTtsRealtimeRegion::Beijing,
            "beijing-api-key",
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime",
        ),
        (
            QwenTtsRealtimeRegion::Singapore,
            "singapore-api-key",
            "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime",
        ),
    ] {
        let (fake, peer) = fake_transport();
        let active_service = make_service(fake, region).unwrap();
        let session = active_service
            .connect(
                &options(api_key),
                &configured_request(QwenTtsRealtimeMode::ServerCommit),
                QwenTtsRealtimeLimits::default(),
            )
            .await
            .unwrap();
        assert_eq!(session.metadata().session_id, "qwen-tts-session-1");
        assert_eq!(session.metadata().created_native["type"], "session.created");
        assert_eq!(session.metadata().updated_native["type"], "session.updated");
        assert!(session.metadata().setup_events.is_empty());

        assert_eq!(peer.connect_count(), 1);
        let connect = peer.connect_request();
        assert_eq!(connect.endpoint, endpoint);
        assert_eq!(
            header(&connect, "authorization").unwrap(),
            format!("Bearer {api_key}")
        );
        assert_eq!(
            header(&connect, "x-dashscope-workspace"),
            Some("workspace-abc123")
        );
        assert!(!format!("{connect:?}").contains(api_key));
        let sent = sent_json(&peer);
        assert_client_event_ids_are_unique_uuids(&sent);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0]["type"], "session.update");
        assert_eq!(sent[0]["session"]["voice"], "Cherry");
        assert_eq!(sent[0]["session"]["mode"], "server_commit");
        assert_eq!(sent[0]["session"]["language_type"], "English");
        assert_eq!(sent[0]["session"]["response_format"], "opus");
        assert_eq!(sent[0]["session"]["sample_rate"], 48000);
        assert_eq!(sent[0]["session"]["speech_rate"], 1.25);
        assert_eq!(sent[0]["session"]["pitch_rate"], 1.1);
        assert_eq!(sent[0]["session"]["volume"], 72);
        assert_eq!(sent[0]["session"]["bit_rate"], 160);
    }
}

#[tokio::test]
async fn missing_key_and_account_mismatch_fail_before_websocket_connect() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let request = make_request(QwenTtsRealtimeMode::ServerCommit);

    assert!(service
        .connect(
            &RequestOptions::default(),
            &request,
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .is_err());
    let wrong_account = RequestOptions {
        account_scope: Some("another-account".into()),
        ..options("beijing-api-key")
    };
    assert!(service
        .connect(&wrong_account, &request, QwenTtsRealtimeLimits::default(),)
        .await
        .is_err());
    assert_eq!(peer.connect_count(), 0);
}

#[tokio::test]
async fn update_ack_is_required_before_connect_returns() {
    let (fake, peer) = fake_transport_with(Vec::new(), false, false, false);
    let mut config = QwenTtsRealtimeConfig::new(
        "qwen-tts-profile",
        "billing-account-7",
        QwenTtsRealtimeRegion::Beijing,
        "workspace-abc123",
    );
    config.connect_timeout = Duration::from_millis(100);
    let service = QwenTtsRealtimeService::new(fake, config).unwrap();

    assert!(service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::Commit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .is_err());
    assert_eq!(peer.connect_count(), 1);
    assert_eq!(sent_json(&peer).len(), 1);
    assert_client_event_ids_are_unique_uuids(&sent_json(&peer));
    assert_eq!(sent_json(&peer)[0]["type"], "session.update");
}

#[tokio::test]
async fn legacy_qwen_tts_is_beijing_pcm_24000_without_qwen3_controls() {
    let legacy = QwenTtsRealtimeRequest::new(
        "qwen-tts-realtime",
        QwenTtsRealtimeVoice::System("Cherry".into()),
    );
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let _session = service
        .connect(
            &options("beijing-api-key"),
            &legacy,
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    assert!(peer
        .connect_request()
        .endpoint
        .contains("?model=qwen-tts-realtime"));
    assert_eq!(sent_json(&peer)[0]["session"]["response_format"], "pcm");
    assert_eq!(sent_json(&peer)[0]["session"]["sample_rate"], 24000);

    let mut opus = legacy.clone();
    opus.format = QwenTtsRealtimeFormat::Opus;
    let mut nondefault_rate = legacy.clone();
    nondefault_rate.sample_rate = 48000;
    let mut speech_rate = legacy.clone();
    speech_rate.speech_rate = Some(1.0);
    let mut volume = legacy.clone();
    volume.volume = Some(50);
    let mut pitch_rate = legacy.clone();
    pitch_rate.pitch_rate = Some(1.0);
    let mut bit_rate = legacy.clone();
    bit_rate.bit_rate = Some(128);
    for invalid in [
        opus,
        nondefault_rate,
        speech_rate,
        volume,
        pitch_rate,
        bit_rate,
    ] {
        assert_preflight_rejected(QwenTtsRealtimeRegion::Beijing, invalid).await;
    }
    assert_preflight_rejected(QwenTtsRealtimeRegion::Singapore, legacy).await;
}

#[tokio::test]
async fn cloning_and_design_models_require_their_matching_voice_origins() {
    for (model, voice) in [
        (
            "qwen3-tts-vc-realtime-2026-01-15",
            QwenTtsRealtimeVoice::System("Cherry".into()),
        ),
        (
            "qwen3-tts-vd-realtime-2026-01-15",
            QwenTtsRealtimeVoice::Cloned("cloned-voice-1".into()),
        ),
        (
            "qwen3-tts-flash-realtime",
            QwenTtsRealtimeVoice::Cloned("cloned-voice-1".into()),
        ),
        (
            "qwen3-tts-flash-realtime",
            QwenTtsRealtimeVoice::Designed("designed-voice-1".into()),
        ),
    ] {
        assert_preflight_rejected(
            QwenTtsRealtimeRegion::Beijing,
            QwenTtsRealtimeRequest::new(model, voice),
        )
        .await;
    }

    for (model, voice) in [
        (
            "qwen3-tts-vc-realtime-2026-01-15",
            QwenTtsRealtimeVoice::Cloned("cloned-voice-1".into()),
        ),
        (
            "qwen3-tts-vd-realtime-2026-01-15",
            QwenTtsRealtimeVoice::Designed("designed-voice-1".into()),
        ),
    ] {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
        let request = QwenTtsRealtimeRequest::new(model, voice);
        let _session = service
            .connect(
                &options("beijing-api-key"),
                &request,
                QwenTtsRealtimeLimits::default(),
            )
            .await
            .unwrap();
        assert_eq!(sent_json(&peer)[0]["session"]["voice"], request.voice.id());
        assert!(peer
            .connect_request()
            .endpoint
            .contains(&format!("?model={model}")));
    }
}

#[tokio::test]
async fn only_instruct_realtime_accepts_instruction_configuration() {
    let mut instruct = QwenTtsRealtimeRequest::new(
        "qwen3-tts-instruct-flash-realtime",
        QwenTtsRealtimeVoice::System("Cherry".into()),
    );
    instruct.instructions = Some("Speak warmly and clearly.".into());
    instruct.optimize_instructions = Some(true);
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let _session = service
        .connect(
            &options("beijing-api-key"),
            &instruct,
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        sent_json(&peer)[0]["session"]["instructions"],
        "Speak warmly and clearly."
    );
    assert_eq!(
        sent_json(&peer)[0]["session"]["optimize_instructions"],
        true
    );

    let mut flash = make_request(QwenTtsRealtimeMode::ServerCommit);
    flash.instructions = instruct.instructions;
    flash.optimize_instructions = instruct.optimize_instructions;
    assert_preflight_rejected(QwenTtsRealtimeRegion::Beijing, flash).await;
}

#[tokio::test]
async fn explicit_language_mismatch_in_session_updated_fails_setup() {
    let (fake, peer) = fake_transport_with_language_mismatch();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let result = service
        .connect(
            &options("beijing-api-key"),
            &configured_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await;
    assert!(result.is_err());
    assert_eq!(peer.connect_count(), 1);
    assert_eq!(sent_json(&peer).len(), 1);
    assert_eq!(sent_json(&peer)[0]["type"], "session.update");
}

#[tokio::test]
async fn omitted_language_in_session_updated_does_not_reject_setup() {
    let (fake, peer) = fake_transport_without_updated_language();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &configured_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        session.metadata().updated_native["session"]["voice"],
        "Cherry"
    );
    assert!(session.metadata().updated_native["session"]
        .get("language_type")
        .is_none());
    assert_eq!(peer.connect_count(), 1);
}

#[tokio::test]
async fn empty_buffer_commit_is_rejected_before_send_in_both_modes() {
    for mode in [
        QwenTtsRealtimeMode::ServerCommit,
        QwenTtsRealtimeMode::Commit,
    ] {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
        let session = service
            .connect(
                &options("beijing-api-key"),
                &make_request(mode),
                QwenTtsRealtimeLimits::default(),
            )
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();

        assert!(input.commit().await.is_err());
        assert_eq!(
            sent_json(&peer)
                .iter()
                .filter(|event| event["type"] == "input_text_buffer.commit")
                .count(),
            0
        );
        input.append_text("clear before commit").await.unwrap();
        input.clear_buffer().await.unwrap();
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            QwenTtsRealtimeEventKind::BufferCleared
        ));
        assert!(input.commit().await.is_err());
        assert_eq!(
            sent_json(&peer)
                .iter()
                .filter(|event| event["type"] == "input_text_buffer.commit")
                .count(),
            0
        );
        input.abort().await.unwrap();
    }
}

#[tokio::test]
async fn commit_and_server_commit_modes_send_only_documented_buffer_events() {
    for (mode, expected_mode) in [
        (QwenTtsRealtimeMode::ServerCommit, "server_commit"),
        (QwenTtsRealtimeMode::Commit, "commit"),
    ] {
        let (fake, peer) = fake_transport();
        let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
        let session = service
            .connect(
                &options("beijing-api-key"),
                &make_request(mode),
                QwenTtsRealtimeLimits::default(),
            )
            .await
            .unwrap();
        let (mut input, mut events) = session.into_parts();
        input
            .append_text("A caller-chosen text chunk.")
            .await
            .unwrap();
        assert!(sent_json(&peer)
            .iter()
            .all(|event| event["type"] != "input_text_buffer.commit"));
        input.commit().await.unwrap();
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            QwenTtsRealtimeEventKind::BufferCommitted { ref item_id }
                if item_id == "user-item-1"
        ));
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            QwenTtsRealtimeEventKind::Native
        ));
        input.clear_buffer().await.unwrap();
        assert!(matches!(
            events.next().await.unwrap().unwrap().kind,
            QwenTtsRealtimeEventKind::BufferCleared
        ));

        let sent = sent_json(&peer);
        assert_client_event_ids_are_unique_uuids(&sent);
        assert_eq!(sent[0]["type"], "session.update");
        assert_eq!(sent[0]["session"]["mode"], expected_mode);
        assert_eq!(sent[1]["type"], "input_text_buffer.append");
        assert_eq!(sent[1]["text"], "A caller-chosen text chunk.");
        assert_eq!(sent[2]["type"], "input_text_buffer.commit");
        assert_eq!(sent[3]["type"], "input_text_buffer.clear");
        assert!(sent.iter().all(|event| event["type"] != "response.create"));
        assert_eq!(peer.connect_count(), 1);
    }
}

#[tokio::test]
async fn clear_buffer_does_not_cancel_audio_already_in_flight() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    input
        .append_text("Text already submitted for synthesis.")
        .await
        .unwrap();

    peer.respond(response_created("resp-1"));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenTtsRealtimeEventKind::Native
    ));
    peer.respond(audio_delta("resp-1", "item-1", "AQID"));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenTtsRealtimeEventKind::AudioDelta { ref data, .. }
            if data == &Bytes::from_static(&[1, 2, 3])
    ));

    input.clear_buffer().await.unwrap();
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenTtsRealtimeEventKind::BufferCleared
    ));
    peer.respond(audio_delta("resp-1", "item-1", "BAUG"));
    let continued_audio = events.next().await.unwrap().unwrap();
    assert!(matches!(
        continued_audio.kind,
        QwenTtsRealtimeEventKind::AudioDelta { ref data, .. }
            if data == &Bytes::from_static(&[4, 5, 6])
    ));
    assert_eq!(continued_audio.native["response_id"], "resp-1");
    assert_eq!(continued_audio.native["item_id"], "item-1");
}

#[tokio::test]
async fn output_audio_ids_usage_and_failed_response_status_are_preserved() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::Commit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (_input, mut events) = session.into_parts();

    peer.respond(response_created("resp-77"));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenTtsRealtimeEventKind::Native
    ));
    peer.respond(audio_delta("resp-77", "item-81", "AAECAw=="));
    let delta = events.next().await.unwrap().unwrap();
    assert!(matches!(
        delta.kind,
        QwenTtsRealtimeEventKind::AudioDelta {
            ref data,
            ref response_id,
            ref item_id,
        } if data == &Bytes::from_static(&[0, 1, 2, 3])
            && response_id == "resp-77"
            && item_id == "item-81"
    ));
    assert_eq!(delta.native["event_id"], "event-audio-delta");

    peer.respond(json!({
        "event_id": "event-audio-done",
        "type": "response.audio.done",
        "response_id": "resp-77",
        "item_id": "item-81",
        "output_index": 0,
        "content_index": 0
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenTtsRealtimeEventKind::AudioDone { ref response_id, ref item_id }
            if response_id == "resp-77" && item_id == "item-81"
    ));

    let completed = response_done(
        "resp-77",
        "completed",
        json!({
            "total_tokens": 67,
            "input_tokens": 3,
            "output_tokens": 64,
            "input_tokens_details": { "text_tokens": 3 },
            "output_tokens_details": { "text_tokens": 0, "audio_tokens": 64 }
        }),
    );
    peer.respond(completed.clone());
    let done = events.next().await.unwrap().unwrap();
    assert!(matches!(
        done.kind,
        QwenTtsRealtimeEventKind::ResponseDone { ref response_id, ref status }
            if response_id == "resp-77" && status == "completed"
    ));
    assert_eq!(done.native["response"]["usage"]["total_tokens"], 67);
    assert_eq!(
        done.native["response"]["usage"]["output_tokens_details"]["audio_tokens"],
        64
    );
    assert_eq!(done.native, completed);

    peer.respond(response_created("resp-78"));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenTtsRealtimeEventKind::Native
    ));
    let failed = response_done(
        "resp-78",
        "failed",
        json!({ "total_tokens": 67, "input_tokens": 3, "output_tokens": 64 }),
    );
    peer.respond(failed.clone());
    let failed_event = events.next().await.unwrap().unwrap();
    assert!(matches!(
        failed_event.kind,
        QwenTtsRealtimeEventKind::ResponseDone { ref response_id, ref status }
            if response_id == "resp-78" && status == "failed"
    ));
    assert_ne!(failed_event.native["response"]["status"], "completed");
    assert_eq!(failed_event.native, failed);
}

#[tokio::test]
async fn finish_requires_ack_then_clean_eof_and_does_not_hide_late_errors() {
    let (fake, peer) = fake_transport_manual_finish();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    input.append_text("final text").await.unwrap();
    input.finish().await.unwrap();
    assert_eq!(sent_json(&peer).last().unwrap()["type"], "session.finish");
    assert_client_event_ids_are_unique_uuids(&sent_json(&peer));

    peer.respond(session_finished());
    let finished = {
        let finishing = events.next();
        tokio::pin!(finishing);
        assert!(futures::poll!(&mut finishing).is_pending());
        peer.close_inbound();
        tokio::time::timeout(Duration::from_millis(200), finishing)
            .await
            .expect("session.finished should complete only after clean EOF")
            .unwrap()
            .unwrap()
    };
    assert!(matches!(
        finished.kind,
        QwenTtsRealtimeEventKind::SessionFinished
    ));
    assert!(events.next().await.unwrap().is_none());

    let (fake, peer) = fake_transport_manual_finish();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    input.finish().await.unwrap();
    peer.respond(session_finished());
    peer.fail_inbound("transport failed after finish acknowledgement");
    assert!(events.next().await.is_err());
    assert_eq!(peer.connect_count(), 1);
}

#[tokio::test]
async fn terminal_event_or_error_survives_canceled_next_during_gated_close() {
    // A clean finish event is held while the injected sink closes. Canceling
    // next() must not consume the terminal event permanently.
    let (fake, peer) = fake_transport_holding_close();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    input.finish().await.unwrap();
    peer.respond(session_finished());
    peer.close_inbound();
    {
        let reading = events.next();
        tokio::pin!(reading);
        assert!(futures::poll!(&mut reading).is_pending());
        assert!(peer.close_pending());
        assert_eq!(peer.eof_poll_count(), 1);
    }
    peer.release_close();
    let finished = tokio::time::timeout(Duration::from_millis(200), events.next())
        .await
        .expect("canceled cleanup must retain the terminal event")
        .unwrap()
        .unwrap();
    assert!(matches!(
        finished.kind,
        QwenTtsRealtimeEventKind::SessionFinished
    ));
    assert!(events.next().await.unwrap().is_none());
    assert_eq!(peer.eof_poll_count(), 1);

    // Provider errors are also returned once after a canceled close, then the
    // stream terminates normally.
    let (fake, peer) = fake_transport_holding_close();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (_input, mut events) = session.into_parts();
    peer.respond(json!({
        "event_id": "event-provider-error",
        "type": "error",
        "error": { "code": "invalid_value", "message": "provider rejected update" }
    }));
    {
        let reading = events.next();
        tokio::pin!(reading);
        assert!(futures::poll!(&mut reading).is_pending());
        assert!(peer.close_pending());
    }
    peer.release_close();
    assert!(
        tokio::time::timeout(Duration::from_millis(200), events.next())
            .await
            .expect("canceled cleanup must retain the provider error")
            .is_err()
    );
    assert!(events.next().await.unwrap().is_none());

    // Failed response.done is a terminal event. Keep the event itself after a
    // canceled sink close, without converting it to success or dropping it.
    let (fake, peer) = fake_transport_holding_close();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (_input, mut events) = session.into_parts();
    peer.respond(response_created("resp-failed"));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenTtsRealtimeEventKind::Native
    ));
    peer.respond(response_done("resp-failed", "failed", json!({})));
    {
        let reading = events.next();
        tokio::pin!(reading);
        assert!(futures::poll!(&mut reading).is_pending());
        assert!(peer.close_pending());
    }
    peer.release_close();
    let failed = tokio::time::timeout(Duration::from_millis(200), events.next())
        .await
        .expect("canceled cleanup must retain failed response.done")
        .unwrap()
        .unwrap();
    assert!(matches!(
        failed.kind,
        QwenTtsRealtimeEventKind::ResponseDone { ref status, .. } if status == "failed"
    ));
    assert!(events.next().await.unwrap().is_none());
}

#[tokio::test]
async fn canceled_reader_after_finish_eof_resumes_without_repolling_eof() {
    let (fake, peer) = fake_transport_holding_finish_send();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    let finishing = input.finish();
    tokio::pin!(finishing);
    assert!(futures::poll!(&mut finishing).is_pending());
    assert!(peer.finish_send_pending());

    peer.respond(session_finished());
    peer.close_inbound();
    {
        let reading = events.next();
        tokio::pin!(reading);
        assert!(futures::poll!(&mut reading).is_pending());
        assert_eq!(peer.eof_poll_count(), 1);
    }

    peer.release_finish_send();
    tokio::time::timeout(Duration::from_millis(200), finishing)
        .await
        .expect("finish send should complete after release")
        .unwrap();
    let finished = tokio::time::timeout(Duration::from_millis(200), events.next())
        .await
        .expect("retained EOF should complete the finish after send confirmation")
        .unwrap()
        .unwrap();
    assert!(matches!(
        finished.kind,
        QwenTtsRealtimeEventKind::SessionFinished
    ));
    assert_eq!(peer.eof_poll_count(), 1);
    assert!(events.next().await.unwrap().is_none());
}

#[tokio::test]
async fn early_eof_is_terminal_and_frame_limit_rejects_without_replay() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let limits = QwenTtsRealtimeLimits {
        max_frame_bytes: 1024,
        ..QwenTtsRealtimeLimits::default()
    };
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::Commit),
            limits,
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    let long_text = "x".repeat(2048);
    assert!(input.append_text(&long_text).await.is_err());
    assert_eq!(
        sent_json(&peer)
            .iter()
            .filter(|event| event["type"] == "input_text_buffer.append")
            .count(),
        0
    );

    peer.close_inbound();
    assert!(events.next().await.is_err());
    assert!(input.append_text("after early EOF").await.is_err());
    assert_eq!(peer.connect_count(), 1);
    assert_eq!(
        sent_json(&peer)
            .iter()
            .filter(|event| event["type"] == "input_text_buffer.append")
            .count(),
        0
    );
}

#[tokio::test]
async fn dropping_pending_append_wakes_reader_and_never_replays_unknown_send() {
    let (fake, peer) = fake_transport_holding_appends();
    let service = make_service(fake, QwenTtsRealtimeRegion::Beijing).unwrap();
    let session = service
        .connect(
            &options("beijing-api-key"),
            &make_request(QwenTtsRealtimeMode::ServerCommit),
            QwenTtsRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    let reading = events.next();
    tokio::pin!(reading);
    assert!(futures::poll!(&mut reading).is_pending());
    {
        let appending = input.append_text("the accepted send may have reached Qwen");
        tokio::pin!(appending);
        assert!(futures::poll!(&mut appending).is_pending());
        assert!(peer.send_pending());
    }
    let read_result = tokio::time::timeout(Duration::from_millis(200), reading)
        .await
        .expect("dropping an in-flight append must wake the event reader");
    assert!(matches!(read_result, Err(_) | Ok(None)));
    assert!(input
        .append_text("do not replay the unknown send")
        .await
        .is_err());
    assert_eq!(
        sent_json(&peer)
            .iter()
            .filter(|event| event["type"] == "input_text_buffer.append")
            .count(),
        1
    );
    assert_client_event_ids_are_unique_uuids(&sent_json(&peer));
    input.abort().await.unwrap();
    assert_eq!(peer.close_count(), 1);
    assert_eq!(peer.connect_count(), 1);
}
