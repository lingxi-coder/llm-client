use async_trait::async_trait;
use futures::{
    channel::{mpsc, oneshot},
    stream::BoxStream,
    Stream, StreamExt,
};
use lingxi_llm_client::{
    client::RequestOptions,
    protocol::Secret,
    qwen_asr_realtime::{
        QwenAsrLanguage, QwenAsrRealtimeAudioFormat, QwenAsrRealtimeConfig, QwenAsrRealtimeError,
        QwenAsrRealtimeEventKind, QwenAsrRealtimeLimits, QwenAsrRealtimeRegion,
        QwenAsrRealtimeRequest, QwenAsrRealtimeService, QwenAsrRealtimeTurnDetection,
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
    inbound_sender: Mutex<Option<mpsc::UnboundedSender<Incoming>>>,
    append_pending: Mutex<bool>,
    close_pending: Mutex<bool>,
    append_release: Mutex<Option<oneshot::Sender<()>>>,
    close_release: Mutex<Option<oneshot::Sender<()>>>,
    omit_updated_sample_rate: Mutex<bool>,
    updated_sample_rate_override: Mutex<Option<u32>>,
    eof_poll_count: Arc<AtomicUsize>,
}

struct FakeTransport {
    shared: Arc<Shared>,
    hold_append: bool,
    hold_close: bool,
    append_release: Mutex<Option<oneshot::Receiver<()>>>,
    close_release: Mutex<Option<oneshot::Receiver<()>>>,
}

struct FakePeer {
    shared: Arc<Shared>,
}

struct FakeSink {
    shared: Arc<Shared>,
    hold_append: bool,
    hold_close: bool,
    append_release: Option<oneshot::Receiver<()>>,
    close_release: Option<oneshot::Receiver<()>>,
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
        assert!(!this.ended, "fake ASR stream polled after EOF");
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
    fake_transport_with_gates(false, false)
}

fn fake_transport_holding_appends() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_gates(true, false)
}

fn fake_transport_holding_close() -> (Arc<FakeTransport>, FakePeer) {
    fake_transport_with_gates(false, true)
}

fn fake_transport_omitting_updated_sample_rate() -> (Arc<FakeTransport>, FakePeer) {
    let (transport, peer) = fake_transport();
    *transport.shared.omit_updated_sample_rate.lock().unwrap() = true;
    (transport, peer)
}

fn fake_transport_with_sample_rate_override(sample_rate: u32) -> (Arc<FakeTransport>, FakePeer) {
    let (transport, peer) = fake_transport();
    *transport
        .shared
        .updated_sample_rate_override
        .lock()
        .unwrap() = Some(sample_rate);
    (transport, peer)
}

fn fake_transport_with_gates(
    hold_append: bool,
    hold_close: bool,
) -> (Arc<FakeTransport>, FakePeer) {
    let (append_release, append_release_rx) = oneshot::channel();
    let (close_release, close_release_rx) = oneshot::channel();
    let shared = Arc::new(Shared {
        outbound: Mutex::new(Vec::new()),
        connect_request: Mutex::new(None),
        connect_count: Mutex::new(0),
        close_count: Mutex::new(0),
        inbound_sender: Mutex::new(None),
        append_pending: Mutex::new(false),
        close_pending: Mutex::new(false),
        append_release: Mutex::new(hold_append.then_some(append_release)),
        close_release: Mutex::new(hold_close.then_some(close_release)),
        omit_updated_sample_rate: Mutex::new(false),
        updated_sample_rate_override: Mutex::new(None),
        eof_poll_count: Arc::new(AtomicUsize::new(0)),
    });
    (
        Arc::new(FakeTransport {
            shared: shared.clone(),
            hold_append,
            hold_close,
            append_release: Mutex::new(hold_append.then_some(append_release_rx)),
            close_release: Mutex::new(hold_close.then_some(close_release_rx)),
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
        *self.shared.connect_request.lock().unwrap() = Some(request.clone());
        *self.shared.connect_count.lock().unwrap() += 1;
        let model = query_model(&request.endpoint).unwrap_or("qwen3-asr-flash-realtime");
        let (sender, receiver) = mpsc::unbounded();
        *self.shared.inbound_sender.lock().unwrap() = Some(sender);
        let initial = vec![Ok(RealtimeFrame::text(session_created(model).to_string()))];
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                shared: self.shared.clone(),
                hold_append: self.hold_append,
                hold_close: self.hold_close,
                append_release: self.append_release.lock().unwrap().take(),
                close_release: self.close_release.lock().unwrap().take(),
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
                message: "Qwen ASR Realtime sends JSON text events".into(),
            });
        };
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|error| RealtimeError::InvalidInput {
                message: format!("outbound event was invalid JSON: {error}"),
            })?;
        self.shared.outbound.lock().unwrap().push(value.clone());

        match value.get("type").and_then(Value::as_str) {
            Some("session.update") => {
                let mut session = value.get("session").cloned().unwrap_or_else(|| json!({}));
                if !session.is_object() {
                    session = json!({});
                }
                session["id"] = json!("qwen-asr-session-1");
                session["object"] = json!("realtime.session");
                session["model"] = json!(self.requested_model());
                session["modalities"] = json!(["text"]);
                if session["input_audio_format"] == "pcm" {
                    // The provider's ASR session.updated sample uses pcm16;
                    // both spellings are accepted for the PCM request.
                    session["input_audio_format"] = json!("pcm16");
                }
                if *self.shared.omit_updated_sample_rate.lock().unwrap() {
                    session.as_object_mut().unwrap().remove("sample_rate");
                } else if let Some(sample_rate) =
                    *self.shared.updated_sample_rate_override.lock().unwrap()
                {
                    session["sample_rate"] = json!(sample_rate);
                }
                self.respond(json!({
                    "event_id": "server-session-updated",
                    "type": "session.updated",
                    "session": session
                }));
            }
            Some("input_audio_buffer.commit") => {
                self.respond(buffer_committed("item-manual-1", None));
            }
            Some("input_audio_buffer.append") if self.hold_append => {
                *self.shared.append_pending.lock().unwrap() = true;
                if let Some(receiver) = self.append_release.take() {
                    let _ = receiver.await;
                }
                *self.shared.append_pending.lock().unwrap() = false;
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
    fn requested_model(&self) -> String {
        self.shared
            .connect_request
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|request| query_model(&request.endpoint))
            .unwrap_or("qwen3-asr-flash-realtime")
            .to_owned()
    }

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

    fn append_pending(&self) -> bool {
        *self.shared.append_pending.lock().unwrap()
    }

    fn release_append(&self) {
        if let Some(sender) = self.shared.append_release.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }

    fn close_pending(&self) -> bool {
        *self.shared.close_pending.lock().unwrap()
    }

    fn release_close(&self) {
        if let Some(sender) = self.shared.close_release.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }

    fn eof_poll_count(&self) -> usize {
        self.shared.eof_poll_count.load(Ordering::SeqCst)
    }
}

fn session_created(model: &str) -> Value {
    json!({
        "event_id": "server-session-created",
        "type": "session.created",
        "session": {
            "id": "qwen-asr-session-1",
            "object": "realtime.session",
            "model": model,
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "input_audio_transcription": null,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.2,
                "silence_duration_ms": 800
            }
        }
    })
}

fn session_finished() -> Value {
    json!({ "event_id": "server-session-finished", "type": "session.finished" })
}

fn buffer_committed(item_id: &str, previous_item_id: Option<&str>) -> Value {
    json!({
        "event_id": "server-buffer-committed",
        "type": "input_audio_buffer.committed",
        "item_id": item_id,
        "previous_item_id": previous_item_id
    })
}

fn make_service(
    transport: Arc<FakeTransport>,
    region: QwenAsrRealtimeRegion,
) -> QwenAsrRealtimeService {
    QwenAsrRealtimeService::new(
        transport,
        QwenAsrRealtimeConfig::new(
            "qwen-asr-profile",
            "speech-account-42",
            region,
            "workspace-abc123",
        ),
    )
    .unwrap()
}

fn options(api_key: &str) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new(api_key.to_owned())),
        account_scope: Some("speech-account-42".into()),
        ..RequestOptions::default()
    }
}

fn request() -> QwenAsrRealtimeRequest {
    QwenAsrRealtimeRequest::new("qwen3-asr-flash-realtime")
}

async fn assert_preflight_rejected(
    region: QwenAsrRealtimeRegion,
    options: &RequestOptions,
    request: &QwenAsrRealtimeRequest,
) {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, region);
    assert!(service
        .connect(options, request, QwenAsrRealtimeLimits::default())
        .await
        .is_err());
    assert_eq!(peer.connect_count(), 0);
    assert!(peer.outbound().is_empty());
}

fn query_model(endpoint: &str) -> Option<&str> {
    endpoint
        .split_once("?model=")
        .map(|(_, model)| model.split('&').next().unwrap_or(model))
}

fn header<'a>(request: &'a RealtimeConnectRequest, name: &str) -> Option<&'a str> {
    request
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
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

#[tokio::test]
async fn workspace_regions_and_credentials_bind_the_websocket_request() {
    for (region, domain) in [
        (QwenAsrRealtimeRegion::Beijing, "cn-beijing"),
        (QwenAsrRealtimeRegion::Singapore, "ap-southeast-1"),
    ] {
        let (fake, peer) = fake_transport();
        let mut config = QwenAsrRealtimeConfig::new(
            "qwen-asr-profile",
            "speech-account-42",
            region,
            "workspace-abc123",
        );
        config.data_inspection = region == QwenAsrRealtimeRegion::Beijing;
        let service = QwenAsrRealtimeService::new(fake, config).unwrap();
        let session = service
            .connect(
                &options("regional-asr-key"),
                &request(),
                QwenAsrRealtimeLimits::default(),
            )
            .await
            .unwrap();
        let connect = peer.connect_request();
        assert_eq!(
            connect.endpoint,
            format!(
                "wss://workspace-abc123.{domain}.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"
            )
        );
        assert_eq!(
            header(&connect, "authorization"),
            Some("Bearer regional-asr-key")
        );
        assert_eq!(
            header(&connect, "X-DashScope-WorkSpace"),
            Some("workspace-abc123")
        );
        assert_eq!(
            header(&connect, "X-DashScope-DataInspection"),
            (region == QwenAsrRealtimeRegion::Beijing).then_some("enable")
        );
        assert_eq!(peer.connect_count(), 1);
        assert_eq!(session.metadata().session_id, "qwen-asr-session-1");
        assert_eq!(session.metadata().model, "qwen3-asr-flash-realtime");
        assert_eq!(session.metadata().scope.region, region);
        assert_eq!(session.metadata().scope.workspace_id, "workspace-abc123");
        assert_client_event_ids_are_unique_uuids(&peer.outbound());
    }
}

#[tokio::test]
async fn account_mismatch_and_invalid_audio_settings_fail_before_websocket() {
    let wrong_account = RequestOptions {
        account_scope: Some("another-billing-account".into()),
        ..options("regional-asr-key")
    };
    assert_preflight_rejected(QwenAsrRealtimeRegion::Beijing, &wrong_account, &request()).await;

    let no_credential = RequestOptions {
        credential: None,
        ..options("regional-asr-key")
    };
    assert_preflight_rejected(QwenAsrRealtimeRegion::Beijing, &no_credential, &request()).await;

    let mut invalid_rate = request();
    invalid_rate.sample_rate = 44_100;
    assert_preflight_rejected(
        QwenAsrRealtimeRegion::Beijing,
        &options("key"),
        &invalid_rate,
    )
    .await;

    let mut different_protocol_model = request();
    different_protocol_model.model = "qwen-audio-3.1-asr-flash-streaming".into();
    assert_preflight_rejected(
        QwenAsrRealtimeRegion::Beijing,
        &options("key"),
        &different_protocol_model,
    )
    .await;

    let mut invalid_vad = request();
    invalid_vad.turn_detection = QwenAsrRealtimeTurnDetection::ServerVad {
        threshold: 1.1,
        silence_duration_ms: 800,
    };
    assert_preflight_rejected(
        QwenAsrRealtimeRegion::Beijing,
        &options("key"),
        &invalid_vad,
    )
    .await;
}

#[tokio::test]
async fn session_updated_may_omit_sample_rate_but_must_not_disagree() {
    let (fake, peer) = fake_transport_omitting_updated_sample_rate();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    assert_eq!(session.metadata().sample_rate, 16_000);
    assert!(peer.outbound()[0]["session"]["sample_rate"]
        .as_u64()
        .is_some());

    let (fake, peer) = fake_transport_with_sample_rate_override(8_000);
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    assert!(service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .is_err());
    assert_eq!(peer.connect_count(), 1);
    assert_eq!(peer.outbound().len(), 1);
}

#[tokio::test]
async fn pcm_and_opus_audio_settings_are_sent_and_audio_bytes_are_base64_encoded() {
    let mut opus_8khz = request();
    opus_8khz.format = QwenAsrRealtimeAudioFormat::Opus;
    opus_8khz.sample_rate = 8000;
    opus_8khz.language = Some(QwenAsrLanguage::English);
    opus_8khz.corpus_text = Some("LingXi, Qwen, and ASR".into());
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("beijing-asr-key"),
            &opus_8khz,
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, _events) = session.into_parts();
    let configured = &peer.outbound()[0]["session"];
    assert_eq!(configured["input_audio_format"], "opus");
    assert_eq!(configured["sample_rate"], 8000);
    assert_eq!(configured["input_audio_transcription"]["language"], "en");
    assert_eq!(
        configured["input_audio_transcription"]["corpus"]["text"],
        "LingXi, Qwen, and ASR"
    );

    let audio = [0x00, 0x01, 0x02, 0xfd, 0xfe, 0xff];
    input.append_audio(&audio).await.unwrap();
    let sent = peer.outbound();
    assert_eq!(sent[1]["type"], "input_audio_buffer.append");
    assert_eq!(sent[1]["audio"], "AAEC/f7/");
    assert_client_event_ids_are_unique_uuids(&sent);
}

#[tokio::test]
async fn vad_transcript_prefix_stash_completion_and_item_failure_are_preserved() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let mut vad_request = request();
    vad_request.turn_detection = QwenAsrRealtimeTurnDetection::ServerVad {
        threshold: 0.0,
        silence_duration_ms: 400,
    };
    let session = service
        .connect(
            &options("asr-key"),
            &vad_request,
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();

    input.append_audio(&[1, 2, 3, 4]).await.unwrap();
    assert_eq!(peer.outbound()[1]["audio"], "AQIDBA==");
    assert!(
        input.commit().await.is_err(),
        "commit is disabled in VAD mode"
    );
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|event| event["type"] == "input_audio_buffer.commit")
            .count(),
        0
    );

    peer.respond(json!({
        "event_id": "server-speech-started",
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": 64,
        "item_id": "item-vad-1"
    }));
    let started = events.next().await.unwrap().unwrap();
    assert!(matches!(
        started.kind,
        QwenAsrRealtimeEventKind::SpeechStarted { ref item_id, audio_start_ms: 64 }
            if item_id == "item-vad-1"
    ));
    assert_eq!(started.native["type"], "input_audio_buffer.speech_started");

    peer.respond(json!({
        "event_id": "server-speech-stopped",
        "type": "input_audio_buffer.speech_stopped",
        "audio_end_ms": 28128,
        "item_id": "item-vad-1"
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::SpeechStopped { ref item_id, audio_end_ms: 28128 }
            if item_id == "item-vad-1"
    ));

    peer.respond(buffer_committed("item-vad-1", Some("item-before")));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::BufferCommitted { ref item_id, ref previous_item_id }
            if item_id == "item-vad-1" && previous_item_id.as_deref() == Some("item-before")
    ));
    peer.respond(json!({
        "type": "conversation.item.created",
        "event_id": "server-item-created",
        "previous_item_id": "item-before",
        "item": {
            "id": "item-vad-1",
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{ "type": "input_audio", "transcript": null }]
        }
    }));
    let created = events.next().await.unwrap().unwrap();
    assert!(
        matches!(created.kind, QwenAsrRealtimeEventKind::ItemCreated { ref item_id } if item_id == "item-vad-1")
    );
    assert_eq!(created.native["item"]["id"], "item-vad-1");

    let partial = json!({
        "event_id": "server-transcription-text",
        "type": "conversation.item.input_audio_transcription.text",
        "item_id": "item-vad-1",
        "content_index": 0,
        "language": "en",
        "emotion": "neutral",
        "text": "The weather is",
        "stash": " nice today"
    });
    peer.respond(partial.clone());
    let preview = events.next().await.unwrap().unwrap();
    assert!(matches!(
        preview.kind,
        QwenAsrRealtimeEventKind::PartialTranscript {
            ref item_id,
            content_index: 0,
            ref text,
            ref stash,
            ref language,
            ref emotion,
        } if item_id == "item-vad-1"
            && text == "The weather is"
            && stash == " nice today"
            && language.as_deref() == Some("en")
            && emotion.as_deref() == Some("neutral")
    ));
    assert_eq!(preview.native, partial);

    let completed = json!({
        "event_id": "server-transcription-completed",
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-vad-1",
        "content_index": 0,
        "language": "en",
        "emotion": "neutral",
        "transcript": "The weather is nice today."
    });
    peer.respond(completed.clone());
    let final_transcript = events.next().await.unwrap().unwrap();
    assert!(matches!(
        final_transcript.kind,
        QwenAsrRealtimeEventKind::TranscriptCompleted {
            ref item_id,
            content_index: 0,
            ref transcript,
            ref language,
            ref emotion,
        } if item_id == "item-vad-1"
            && transcript == "The weather is nice today."
            && language.as_deref() == Some("en")
            && emotion.as_deref() == Some("neutral")
    ));
    assert_eq!(final_transcript.native, completed);

    input.append_audio(&[5, 6, 7, 8]).await.unwrap();
    peer.respond(buffer_committed("item-vad-2", Some("item-vad-1")));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::BufferCommitted { ref item_id, .. }
            if item_id == "item-vad-2"
    ));
    peer.respond(json!({
        "event_id": "server-item-created-2",
        "type": "conversation.item.created",
        "previous_item_id": "item-vad-1",
        "item": {
            "id": "item-vad-2",
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{ "type": "input_audio", "transcript": null }]
        }
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::ItemCreated { ref item_id }
            if item_id == "item-vad-2"
    ));
    let failed = json!({
        "event_id": "server-transcription-failed",
        "type": "conversation.item.input_audio_transcription.failed",
        "item_id": "item-vad-2",
        "content_index": 0,
        "error": {
            "code": "audio_decode_failed",
            "message": "The audio chunk could not be decoded.",
            "param": "audio"
        }
    });
    peer.respond(failed.clone());
    let failure = events.next().await.unwrap().unwrap();
    assert!(matches!(
        failure.kind,
        QwenAsrRealtimeEventKind::TranscriptFailed {
            ref item_id,
            content_index: 0,
            ref code,
            ref message,
        } if item_id == "item-vad-2"
            && code.as_deref() == Some("audio_decode_failed")
            && message.as_deref() == Some("The audio chunk could not be decoded.")
    ));
    assert_eq!(failure.native, failed);
    assert_eq!(
        peer.close_count(),
        0,
        "an item failure does not close the session"
    );

    // The failure is scoped to its item. A later utterance can still be
    // recognized, and the session can complete normally afterward.
    input.append_audio(&[9, 10, 11, 12]).await.unwrap();
    peer.respond(buffer_committed("item-vad-3", Some("item-vad-2")));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::BufferCommitted { ref item_id, .. }
            if item_id == "item-vad-3"
    ));
    peer.respond(json!({
        "event_id": "server-item-created-3",
        "type": "conversation.item.created",
        "previous_item_id": "item-vad-2",
        "item": {
            "id": "item-vad-3",
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{ "type": "input_audio", "transcript": null }]
        }
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::ItemCreated { ref item_id }
            if item_id == "item-vad-3"
    ));
    peer.respond(json!({
        "event_id": "server-transcription-completed-3",
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-vad-3",
        "content_index": 0,
        "language": "en",
        "transcript": "The next utterance succeeded."
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::TranscriptCompleted { ref item_id, ref transcript, .. }
            if item_id == "item-vad-3" && transcript == "The next utterance succeeded."
    ));
    input.finish().await.unwrap();
    peer.respond(session_finished());
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::SessionFinished
    ));
    assert_eq!(peer.close_count(), 1);
    assert_eq!(peer.eof_poll_count(), 1);
}

#[tokio::test]
async fn manual_mode_commits_audio_and_vad_mode_never_emits_commit() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let mut manual = request();
    manual.turn_detection = QwenAsrRealtimeTurnDetection::Manual;
    let session = service
        .connect(
            &options("asr-key"),
            &manual,
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    assert!(peer.outbound()[0]["session"]["turn_detection"].is_null());
    input.append_audio(&[0x10, 0x20, 0x30, 0x40]).await.unwrap();
    assert!(
        input.finish().await.is_err(),
        "manual mode cannot finish uncommitted audio"
    );
    input.commit().await.unwrap();
    assert_eq!(peer.outbound()[2]["type"], "input_audio_buffer.commit");
    // The documented protocol permits sending session.finish after commit
    // without first draining the commit acknowledgement.
    input.finish().await.unwrap();
    assert_eq!(peer.outbound().last().unwrap()["type"], "session.finish");
    assert_client_event_ids_are_unique_uuids(&peer.outbound());
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::BufferCommitted { ref item_id, .. }
            if item_id == "item-manual-1"
    ));
    peer.respond(json!({
        "event_id": "server-manual-item-created",
        "type": "conversation.item.created",
        "previous_item_id": null,
        "item": {
            "id": "item-manual-1",
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{ "type": "input_audio", "transcript": null }]
        }
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::ItemCreated { ref item_id }
            if item_id == "item-manual-1"
    ));

    peer.respond(json!({
        "event_id": "server-manual-completed",
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-manual-1",
        "content_index": 0,
        "transcript": "Turn ended."
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::TranscriptCompleted { ref transcript, .. }
            if transcript == "Turn ended."
    ));
    peer.respond(session_finished());
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::SessionFinished
    ));
    assert_eq!(peer.close_count(), 1);
}

#[tokio::test]
async fn finish_requires_final_transcript_session_finished_and_clean_eof() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    input.finish().await.unwrap();
    assert_eq!(peer.outbound().last().unwrap()["type"], "session.finish");

    peer.respond(buffer_committed("item-finish", None));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::BufferCommitted { ref item_id, .. }
            if item_id == "item-finish"
    ));
    peer.respond(json!({
        "event_id": "server-finish-item-created",
        "type": "conversation.item.created",
        "previous_item_id": null,
        "item": {
            "id": "item-finish",
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{ "type": "input_audio", "transcript": null }]
        }
    }));
    assert!(matches!(
        events.next().await.unwrap().unwrap().kind,
        QwenAsrRealtimeEventKind::ItemCreated { ref item_id }
            if item_id == "item-finish"
    ));
    peer.respond(json!({
        "event_id": "server-final-transcript",
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-finish",
        "content_index": 0,
        "language": "en",
        "transcript": "Final recognition before finish."
    }));
    peer.respond(session_finished());
    let transcript = events.next().await.unwrap().unwrap();
    assert!(matches!(
        transcript.kind,
        QwenAsrRealtimeEventKind::TranscriptCompleted { .. }
    ));
    let finish = tokio::time::timeout(Duration::from_millis(250), events.next())
        .await
        .expect("session.finish should close locally and wait for EOF")
        .unwrap()
        .unwrap();
    assert!(matches!(
        finish.kind,
        QwenAsrRealtimeEventKind::SessionFinished
    ));
    assert_eq!(peer.close_count(), 1);
    assert_eq!(peer.eof_poll_count(), 1);
    assert!(events.next().await.unwrap().is_none());
    assert_eq!(peer.eof_poll_count(), 1);
    assert_client_event_ids_are_unique_uuids(&peer.outbound());
}

#[tokio::test]
async fn late_error_after_session_finished_prevents_clean_success() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    input.finish().await.unwrap();
    peer.respond(session_finished());
    peer.fail_inbound("late socket error after session.finished");
    peer.close_inbound();

    let result = tokio::time::timeout(Duration::from_millis(250), events.next())
        .await
        .expect("late inbound error should not be hidden by finish acknowledgement");
    assert!(matches!(
        result,
        Err(QwenAsrRealtimeError::Interrupted { .. }) | Err(QwenAsrRealtimeError::Realtime(_))
    ));
    assert_eq!(peer.connect_count(), 1);
    assert_eq!(peer.outbound().last().unwrap()["type"], "session.finish");
}

#[tokio::test]
async fn early_eof_is_interrupted_and_does_not_replay_or_report_finished() {
    let (fake, peer) = fake_transport();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    peer.close_inbound();
    assert!(events.next().await.is_err());
    assert!(input.append_audio(&[1, 2]).await.is_err());
    assert_eq!(peer.eof_poll_count(), 1);
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|event| event["type"] == "input_audio_buffer.append")
            .count(),
        0
    );
    assert_eq!(peer.close_count(), 1);
    assert!(events.next().await.unwrap().is_none());
    assert_eq!(peer.eof_poll_count(), 1);
}

#[tokio::test]
async fn dropping_pending_audio_write_wakes_reader_and_never_replays() {
    let (fake, peer) = fake_transport_holding_appends();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    let reading = events.next();
    tokio::pin!(reading);
    assert!(futures::poll!(&mut reading).is_pending());
    {
        let appending = input.append_audio(&[1, 2, 3, 4]);
        tokio::pin!(appending);
        assert!(futures::poll!(&mut appending).is_pending());
        assert!(peer.append_pending());
    }
    let read_result = tokio::time::timeout(Duration::from_millis(250), reading)
        .await
        .expect("dropping append during an in-flight send must notify events");
    assert!(matches!(read_result, Err(_) | Ok(None)));
    assert!(input.append_audio(&[1, 2, 3, 4]).await.is_err());
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|event| event["type"] == "input_audio_buffer.append")
            .count(),
        1
    );
    input.abort().await.unwrap();
}

#[tokio::test]
async fn audio_write_waits_for_transport_backpressure_before_reporting_success() {
    let (fake, peer) = fake_transport_holding_appends();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, _events) = session.into_parts();
    {
        let append = input.append_audio(&[0x12, 0x34]);
        tokio::pin!(append);
        assert!(futures::poll!(&mut append).is_pending());
        assert!(peer.append_pending());
        peer.release_append();
        tokio::time::timeout(Duration::from_millis(250), append)
            .await
            .expect("append should complete when the injected writer accepts the frame")
            .unwrap();
    }
    assert_eq!(input.audio_bytes_sent(), 2);
    assert_eq!(
        peer.outbound()
            .iter()
            .filter(|event| event["type"] == "input_audio_buffer.append")
            .count(),
        1
    );
}

#[tokio::test]
async fn canceling_finish_reader_during_close_retains_terminal_once_without_repolling_eof() {
    let (fake, peer) = fake_transport_holding_close();
    let service = make_service(fake, QwenAsrRealtimeRegion::Beijing);
    let session = service
        .connect(
            &options("asr-key"),
            &request(),
            QwenAsrRealtimeLimits::default(),
        )
        .await
        .unwrap();
    let (mut input, mut events) = session.into_parts();
    input.finish().await.unwrap();
    peer.respond(session_finished());
    {
        let reading = events.next();
        tokio::pin!(reading);
        assert!(futures::poll!(&mut reading).is_pending());
        assert!(peer.close_pending());
    }
    peer.release_close();
    let finished = tokio::time::timeout(Duration::from_millis(250), events.next())
        .await
        .expect("canceled close cleanup should retain session.finished")
        .unwrap()
        .unwrap();
    assert!(matches!(
        finished.kind,
        QwenAsrRealtimeEventKind::SessionFinished
    ));
    assert_eq!(peer.close_count(), 1);
    assert_eq!(peer.eof_poll_count(), 1);
    assert!(events.next().await.unwrap().is_none());
    assert_eq!(peer.eof_poll_count(), 1);
}
