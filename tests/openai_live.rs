use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::mpsc,
    executor::block_on,
    stream::{self, BoxStream},
    Stream, StreamExt,
};
use lingxi_llm_client::providers::openai::live::OpenAiLiveAudioFormat;
use lingxi_llm_client::providers::openai::live::OpenAiLiveCommand;
use lingxi_llm_client::providers::openai::live::OpenAiLiveConfig;
use lingxi_llm_client::providers::openai::live::OpenAiLiveDelegation;
use lingxi_llm_client::providers::openai::live::OpenAiLiveEvent;
use lingxi_llm_client::providers::openai::live::OpenAiLiveHistoryMessage;
use lingxi_llm_client::providers::openai::live::OpenAiLiveHistoryRole;
use lingxi_llm_client::providers::openai::live::OpenAiLiveResponsesConfig;
use lingxi_llm_client::providers::openai::live::OpenAiLiveResponsesTool;
use lingxi_llm_client::providers::openai::live::OpenAiLiveResponsesUpdate;
use lingxi_llm_client::providers::openai::live::OpenAiLiveRoute;
use lingxi_llm_client::providers::openai::live::OpenAiLiveScope;
use lingxi_llm_client::providers::openai::live::OpenAiLiveSession;
use lingxi_llm_client::providers::openai::live::OpenAiLiveToolChoice;
use lingxi_llm_client::providers::openai::live::OPENAI_LIVE_WEBSOCKET_ENDPOINT;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeLimits, RealtimeSink, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

type TerminalPollHook = Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>;
type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
    outgoing: mpsc::UnboundedSender<RealtimeFrame>,
    closed: Arc<Mutex<Vec<RealtimeClose>>>,
    terminal_poll_hook: TerminalPollHook,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
}

struct FakePeer {
    incoming: mpsc::UnboundedSender<Incoming>,
    outgoing: mpsc::UnboundedReceiver<RealtimeFrame>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
    closed: Arc<Mutex<Vec<RealtimeClose>>>,
    terminal_poll_hook: TerminalPollHook,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
}

struct FakeSink {
    outgoing: mpsc::UnboundedSender<RealtimeFrame>,
    closed: Arc<Mutex<Vec<RealtimeClose>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
}

struct HookedIncoming {
    inner: BoxStream<'static, Incoming>,
    terminal_poll_hook: TerminalPollHook,
}

impl Stream for HookedIncoming {
    type Item = Incoming;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = this.inner.as_mut().poll_next(cx);
        if let Poll::Ready(Some(Ok(RealtimeFrame::Text(bytes)))) = &item {
            let is_terminal = serde_json::from_slice::<Value>(bytes)
                .ok()
                .and_then(|event| event.get("type").and_then(Value::as_str).map(str::to_owned))
                .is_some_and(|event_type| event_type == "session.closed");
            if is_terminal {
                if let Some(hook) = this.terminal_poll_hook.lock().unwrap().take() {
                    hook();
                }
            }
        }
        item
    }
}

fn fake_transport(preface: &[Value]) -> (Arc<FakeTransport>, FakePeer) {
    let (incoming_tx, incoming_rx) = mpsc::unbounded();
    let prefix = preface
        .iter()
        .map(|value| Ok(RealtimeFrame::text(value.to_string())))
        .collect::<Vec<_>>();
    let incoming = stream::iter(prefix).chain(incoming_rx).boxed();
    let (outgoing_tx, outgoing_rx) = mpsc::unbounded();
    let request = Arc::new(Mutex::new(None));
    let connect_count = Arc::new(Mutex::new(0));
    let closed = Arc::new(Mutex::new(Vec::new()));
    let terminal_poll_hook = Arc::new(Mutex::new(None));
    let sent = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(incoming)),
            request: request.clone(),
            connect_count: connect_count.clone(),
            outgoing: outgoing_tx,
            closed: closed.clone(),
            terminal_poll_hook: terminal_poll_hook.clone(),
            sent: sent.clone(),
        }),
        FakePeer {
            incoming: incoming_tx,
            outgoing: outgoing_rx,
            request,
            connect_count,
            closed,
            terminal_poll_hook,
            sent,
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
            .expect("one fake connection");
        let incoming = HookedIncoming {
            inner: incoming,
            terminal_poll_hook: self.terminal_poll_hook.clone(),
        }
        .boxed();
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                outgoing: self.outgoing.clone(),
                closed: self.closed.clone(),
                sent: self.sent.clone(),
            }),
            inbound: incoming,
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.sent.lock().unwrap().push(frame.clone());
        self.outgoing
            .unbounded_send(frame)
            .map_err(|_| RealtimeError::Closed)
    }

    async fn close(&mut self, close: RealtimeClose) -> Result<(), RealtimeError> {
        self.closed.lock().unwrap().push(close);
        Ok(())
    }
}

fn route() -> OpenAiLiveRoute {
    OpenAiLiveRoute::new()
}

fn scope(route: &OpenAiLiveRoute) -> OpenAiLiveScope {
    OpenAiLiveScope::new("openai-main", "account-9", route).unwrap()
}

fn limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 32,
        max_frame_bytes: 256 * 1024,
    }
}

fn started(model: &str) -> Value {
    json!({
        "type":"session.started",
        "event_id":"evt_started",
        "session":{"id":"live_1","model":model,"status":"active"}
    })
}

fn started_without_model() -> Value {
    json!({
        "type":"session.started",
        "event_id":"evt_started",
        "session":{"id":"live_1","status":"active"}
    })
}

async fn connect_with(
    config: OpenAiLiveConfig,
    preface: &[Value],
) -> Result<
    (
        lingxi_llm_client::providers::openai::live::OpenAiLiveSession,
        lingxi_llm_client::providers::openai::live::OpenAiLiveDriver,
        Arc<FakeTransport>,
        FakePeer,
    ),
    RealtimeError,
> {
    let route = route();
    let (transport, peer) = fake_transport(preface);
    let (session, driver) = OpenAiLiveSession::connect(
        transport.clone(),
        route.clone(),
        scope(&route),
        Secret::from("api-key-test".to_owned()),
        Some("hashed-user-42".into()),
        config,
        limits(),
    )
    .await?;
    Ok((session, driver, transport, peer))
}

async fn next_json(peer: &mut FakePeer) -> Value {
    let frame = peer.outgoing.next().await.expect("outgoing frame");
    match frame {
        RealtimeFrame::Text(bytes) => serde_json::from_slice(&bytes).unwrap(),
        RealtimeFrame::Binary(_) => panic!("GPT-Live primary events are JSON text"),
    }
}

fn json_frame(value: Value) -> Incoming {
    Ok(RealtimeFrame::text(value.to_string()))
}

#[test]
fn connects_to_query_free_primary_endpoint_and_waits_for_started() {
    block_on(async {
        let (session, _driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let start = next_json(&mut peer).await;
        assert_eq!(start["type"], "session.start");
        assert_eq!(start["session"]["model"], "gpt-live-1");
        assert!(start["session"].get("input").is_none());
        assert!(start["session"].get("store").is_none());
        assert_eq!(
            start["session"]["audio"]["format"],
            json!({"type":"audio/pcm","rate":24000})
        );

        let request = peer.request.lock().unwrap().clone().unwrap();
        assert_eq!(request.endpoint, OPENAI_LIVE_WEBSOCKET_ENDPOINT);
        assert!(!request.endpoint.contains('?'));
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer api-key-test"));
        assert!(request.headers.iter().any(|(name, value)| {
            name == "OpenAI-Safety-Identifier" && value == "hashed-user-42"
        }));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);

        let (_control, _events) = session.into_parts();
    });
}

#[test]
fn startup_history_uses_documented_text_parts_and_storage_is_opt_in() {
    block_on(async {
        let config = OpenAiLiveConfig {
            input: vec![
                OpenAiLiveHistoryMessage::new(
                    OpenAiLiveHistoryRole::Developer,
                    "Use the verified account context.",
                ),
                OpenAiLiveHistoryMessage::new(
                    OpenAiLiveHistoryRole::User,
                    "I need help with my recent order.",
                ),
                OpenAiLiveHistoryMessage::new(
                    OpenAiLiveHistoryRole::Assistant,
                    "What is the order number?",
                ),
            ],
            store: true,
            ..OpenAiLiveConfig::default()
        };
        let (session, _driver, _transport, mut peer) =
            connect_with(config, &[started("gpt-live-1")])
                .await
                .unwrap();
        let start = next_json(&mut peer).await;
        assert_eq!(
            start["session"]["input"],
            json!([
                {
                    "type":"message",
                    "role":"developer",
                    "content":[{"type":"input_text","text":"Use the verified account context."}]
                },
                {
                    "type":"message",
                    "role":"user",
                    "content":[{"type":"input_text","text":"I need help with my recent order."}]
                },
                {
                    "type":"message",
                    "role":"assistant",
                    "content":[{"type":"output_text","text":"What is the order number?"}]
                }
            ])
        );
        assert_eq!(start["session"]["store"], true);
        let (_control, _events) = session.into_parts();
    });
}

#[test]
fn startup_history_rejects_more_than_128_messages_before_connecting() {
    block_on(async {
        let (transport, peer) = fake_transport(&[started("gpt-live-1")]);
        let route = route();
        let config = OpenAiLiveConfig {
            input: (0..129)
                .map(|index| {
                    OpenAiLiveHistoryMessage::new(
                        OpenAiLiveHistoryRole::User,
                        format!("prior message {index}"),
                    )
                })
                .collect(),
            ..OpenAiLiveConfig::default()
        };
        let result = OpenAiLiveSession::connect(
            transport,
            route.clone(),
            scope(&route),
            Secret::from("api-key-test".to_owned()),
            None,
            config,
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
    });
}

#[test]
fn invalid_config_fails_before_opening_the_transport() {
    block_on(async {
        let (transport, peer) = fake_transport(&[started("gpt-live-1")]);
        let route = route();
        let config = OpenAiLiveConfig {
            delegation: OpenAiLiveDelegation::Responses(OpenAiLiveResponsesConfig::new("  ")),
            ..OpenAiLiveConfig::default()
        };
        let result = OpenAiLiveSession::connect(
            transport,
            route.clone(),
            scope(&route),
            Secret::from("api-key-test".to_owned()),
            Some("invalid\nheader".into()),
            config,
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);
        assert!(peer.request.lock().unwrap().is_none());
    });
}

#[test]
fn handshake_rejects_a_different_live_model() {
    block_on(async {
        let result =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-realtime-2.1")]).await;
        assert!(matches!(result, Err(RealtimeError::Transport { .. })));
    });
}

#[test]
fn handshake_rejects_a_missing_required_model() {
    block_on(async {
        let result = connect_with(OpenAiLiveConfig::default(), &[started_without_model()]).await;
        assert!(matches!(result, Err(RealtimeError::Transport { .. })));
    });
}

#[test]
fn audio_wire_format_is_fixed_for_both_directions_and_validates_raw_pcm() {
    block_on(async {
        let formats = [
            (
                OpenAiLiveAudioFormat::Pcm16Mono24Khz,
                json!({"type":"audio/pcm","rate":24000}),
                Bytes::from_static(&[1, 2]),
            ),
            (
                OpenAiLiveAudioFormat::Pcm16Mono16Khz,
                json!({"type":"audio/pcm","rate":16000}),
                Bytes::from_static(&[3, 4]),
            ),
            (
                OpenAiLiveAudioFormat::G711MuLaw8Khz,
                json!({"type":"audio/pcmu","rate":8000}),
                Bytes::from_static(&[5]),
            ),
            (
                OpenAiLiveAudioFormat::G711ALaw8Khz,
                json!({"type":"audio/pcma","rate":8000}),
                Bytes::from_static(&[6]),
            ),
        ];

        for (format, expected, audio) in formats {
            let config = OpenAiLiveConfig {
                audio_format: format,
                ..OpenAiLiveConfig::default()
            };
            let (session, driver, _transport, mut peer) =
                connect_with(config, &[started("gpt-live-1")])
                    .await
                    .unwrap();
            let start = next_json(&mut peer).await;
            assert_eq!(start["session"]["audio"]["format"], expected);
            let (control, _events) = session.into_parts();
            if matches!(
                format,
                OpenAiLiveAudioFormat::Pcm16Mono24Khz | OpenAiLiveAudioFormat::Pcm16Mono16Khz
            ) {
                assert!(matches!(
                    control.append_audio(Bytes::from_static(&[0]), None),
                    Err(RealtimeError::InvalidInput { .. })
                ));
            }
            control
                .append_audio(audio.clone(), Some("audio_chunk".into()))
                .unwrap();

            let app = async {
                let command = next_json(&mut peer).await;
                assert_eq!(command["type"], "session.input_audio.append");
                assert_eq!(command["event_id"], "audio_chunk");
                control
                    .abort(RealtimeClose::normal("test cleanup"))
                    .await
                    .unwrap();
            };
            let (driver_result, ()) = futures::join!(driver.run(), app);
            driver_result.unwrap();
        }
    });
}

#[test]
fn responses_items_outputs_and_continuation_are_explicit_caller_commands() {
    block_on(async {
        let mut responses = OpenAiLiveResponsesConfig::new("gpt-6-luna");
        responses.instructions = Some("Check the request carefully.".into());
        responses.tools = vec![
            OpenAiLiveResponsesTool::function(
                "lookup_order",
                json!({"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}),
            )
            .with_description("Look up an order by ID.")
            .unwrap(),
            OpenAiLiveResponsesTool::WebSearch,
        ];
        responses.tool_choice = Some(OpenAiLiveToolChoice::Function("lookup_order".into()));
        responses.parallel_tool_calls = Some(false);
        responses.max_output_tokens = Some(128);
        let config = OpenAiLiveConfig {
            delegation: OpenAiLiveDelegation::Responses(responses),
            ..OpenAiLiveConfig::default()
        };
        let (session, driver, _transport, mut peer) =
            connect_with(config, &[started("gpt-live-1")])
                .await
                .unwrap();
        let start = next_json(&mut peer).await;
        assert_eq!(start["session"]["delegation"]["type"], "responses");
        assert_eq!(
            start["session"]["delegation"]["responses"]["tools"][0]["name"],
            "lookup_order"
        );
        assert_eq!(
            start["session"]["delegation"]["responses"]["tools"][1]["type"],
            "web_search"
        );
        assert_eq!(
            start["session"]["delegation"]["responses"]["tool_choice"],
            json!({"type":"function","name":"lookup_order"})
        );
        let (control, _events) = session.into_parts();

        let app = async {
            control
                .append_responses_text("Where is order A-17?", Some("item_1".into()))
                .unwrap();
            let item = next_json(&mut peer).await;
            assert_eq!(item["type"], "response.item.create");
            assert_eq!(item["item"]["role"], "user");
            assert_eq!(item["item"]["content"][0]["text"], "Where is order A-17?");

            control
                .function_output("call_1", "{\"status\":\"shipped\"}", Some("item_2".into()))
                .unwrap();
            let output = next_json(&mut peer).await;
            assert_eq!(output["type"], "response.item.create");
            assert_eq!(output["item"]["type"], "function_call_output");
            assert_eq!(output["item"]["call_id"], "call_1");

            control
                .continue_responses(Some("continue_1".into()))
                .unwrap();
            let continuation = next_json(&mut peer).await;
            assert_eq!(
                continuation,
                json!({"type":"response.create","event_id":"continue_1"})
            );

            let update = OpenAiLiveResponsesUpdate {
                max_output_tokens: Some(256),
                ..OpenAiLiveResponsesUpdate::default()
            };
            control
                .update_responses(update, Some("update_1".into()))
                .unwrap();
            let update = next_json(&mut peer).await;
            assert_eq!(update["type"], "session.update");
            assert_eq!(
                update["session"]["delegation"]["responses"]["max_output_tokens"],
                256
            );
            control
                .abort(RealtimeClose::normal("test cleanup"))
                .await
                .unwrap();
        };
        let (driver_result, ()) = futures::join!(driver.run(), app);
        driver_result.unwrap();
    });
}

#[test]
fn response_only_commands_are_rejected_in_client_delegation() {
    block_on(async {
        let (session, _driver, _transport, _peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let (control, _events) = session.into_parts();
        assert!(matches!(
            control.send(OpenAiLiveCommand::AppendResponsesText {
                text: "hello".into(),
                event_id: None,
            }),
            Err(RealtimeError::InvalidInput { .. })
        ));
        assert!(matches!(
            control.send(OpenAiLiveCommand::ContinueResponses { event_id: None }),
            Err(RealtimeError::InvalidInput { .. })
        ));
    });
}

#[test]
fn client_context_updates_carry_nullable_or_existing_delegation_ids() {
    block_on(async {
        let (session, driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let _ = next_json(&mut peer).await;
        let (control, mut events) = session.into_parts();
        let app = async {
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::SessionStarted { .. })
            ));
            control
                .append_instructions(
                    "Keep the answer concise.",
                    None,
                    Some("instruction_1".into()),
                )
                .unwrap();
            let instructions = next_json(&mut peer).await;
            assert_eq!(instructions["type"], "session.instructions.append");
            assert!(instructions["delegation_id"].is_null());

            control
                .append_thinking(
                    "Checking the verified account record.",
                    Some("del_7".into()),
                    None,
                )
                .unwrap();
            let thinking = next_json(&mut peer).await;
            assert_eq!(thinking["type"], "session.thinking.append");
            assert_eq!(thinking["delegation_id"], "del_7");

            control
                .append_commentary(
                    "The verified record says it shipped.",
                    Some("del_7".into()),
                    None,
                )
                .unwrap();
            let commentary = next_json(&mut peer).await;
            assert_eq!(commentary["type"], "session.commentary.append");
            assert_eq!(commentary["delegation_id"], "del_7");
            control
                .abort(RealtimeClose::normal("test cleanup"))
                .await
                .unwrap();
        };
        let (driver_result, ()) = futures::join!(driver.run(), app);
        driver_result.unwrap();
    });
}

#[test]
fn primary_events_preserve_audio_timestamps_usage_and_nested_responses_payloads() {
    block_on(async {
        let (session, driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let _ = next_json(&mut peer).await;
        let (control, mut events) = session.into_parts();
        let app = async {
            for value in [
                json!({"type":"session.input_transcript.delta","delta":"I need ","start_ms":2,"end_ms":5}),
                json!({"type":"session.output_transcript.delta","delta":"Hello.","start_ms":6,"end_ms":10}),
                json!({"type":"session.output_audio.delta","delta":"AQI="}),
                json!({"type":"session.delegation.created","offset_ms":12,"delegation":{"id":"del_1","target":"responses","response_id":"resp_1"}}),
                json!({"type":"response.event","delegation_id":"del_1","event":{"type":"response.output_text.delta","delta":"Found it","sequence_number":4}}),
                json!({"type":"session.usage.updated","usage":{"seconds":3.5},"context_window":{"usage_ratio":0.2}}),
                json!({"type":"session.closed","reason":"close_requested","session":{"id":"live_1"},"usage":{"seconds":4.0}}),
            ] {
                peer.incoming.unbounded_send(json_frame(value)).unwrap();
            }

            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::SessionStarted { .. })
            ));
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::InputTranscriptDelta { delta, start_ms: Some(2.0), end_ms: Some(5.0), .. }) if delta == "I need "
            ));
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::OutputTranscriptDelta { delta, .. }) if delta == "Hello."
            ));
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::OutputAudioDelta { data, format: OpenAiLiveAudioFormat::Pcm16Mono24Khz, .. }) if data == Bytes::from_static(&[1, 2])
            ));
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::DelegationCreated { delegation_id: Some(id), target: Some(target), response_id: Some(response_id), offset_ms: Some(12.0), .. }) if id == "del_1" && target == "responses" && response_id == "resp_1"
            ));
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::ResponseEvent { delegation_id: Some(id), event, .. }) if id == "del_1" && event["sequence_number"] == 4
            ));
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::UsageUpdated { seconds: Some(3.5), context_window: Some(window), .. }) if window["usage_ratio"] == 0.2
            ));
            assert!(
                matches!(events.next().await, Some(OpenAiLiveEvent::SessionClosed { reason: Some(reason), usage, .. }) if reason == "close_requested" && usage["seconds"] == 4.0)
            );
            control
                .abort(RealtimeClose::normal("test cleanup"))
                .await
                .unwrap();
        };
        let (driver_result, ()) = futures::join!(driver.run(), app);
        driver_result.unwrap();
    });
}

#[test]
fn session_close_waits_for_terminal_usage_before_transport_close() {
    block_on(async {
        let (session, driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let _ = next_json(&mut peer).await;
        let (control, mut events) = session.into_parts();
        let app = async {
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::SessionStarted { .. })
            ));
            control.request_close(Some("close_1".into())).unwrap();
            assert!(control
                .append_audio(Bytes::from_static(&[1, 2]), None)
                .is_err());
            assert_eq!(
                next_json(&mut peer).await,
                json!({"type":"session.close","event_id":"close_1"})
            );
            assert!(peer.closed.lock().unwrap().is_empty());
            peer.incoming
                .unbounded_send(json_frame(json!({
                    "type":"session.closed",
                    "reason":"close_requested",
                    "session":{"id":"live_1"},
                    "usage":{"seconds":9.5}
                })))
                .unwrap();
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::SessionClosed { usage, .. }) if usage["seconds"] == 9.5
            ));
            assert!(peer.closed.lock().unwrap().is_empty());
            control
                .close_after_session_closed(RealtimeClose::normal("finalized"))
                .await
                .unwrap();
            assert_eq!(
                peer.closed.lock().unwrap().as_slice(),
                &[RealtimeClose::normal("finalized")]
            );
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::TransportClosed { .. })
            ));
        };
        let (driver_result, ()) = futures::join!(driver.run(), app);
        driver_result.unwrap();
    });
}

#[test]
fn a_command_queued_during_terminal_dispatch_is_dropped_without_sending() {
    block_on(async {
        let (session, driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let _ = next_json(&mut peer).await;
        let (control, mut events) = session.into_parts();
        let queued = Arc::new(Mutex::new(false));
        let queued_from_hook = queued.clone();
        let control_from_hook = control.clone();
        *peer.terminal_poll_hook.lock().unwrap() = Some(Box::new(move || {
            *queued_from_hook.lock().unwrap() = control_from_hook
                .append_audio(Bytes::from_static(&[1, 2]), Some("late_audio".into()))
                .is_ok();
        }));

        let app = async {
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::SessionStarted { .. })
            ));
            peer.incoming
                .unbounded_send(json_frame(json!({
                    "type":"session.closed",
                    "reason":"remote_hangup",
                    "session":{"id":"live_1"},
                    "usage":{"seconds":1.0}
                })))
                .unwrap();
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::SessionClosed { .. })
            ));
            assert!(*queued.lock().unwrap());
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::CommandDroppedAfterSessionClosed { event_id: Some(id) }) if id == "late_audio"
            ));
            assert_eq!(peer.sent.lock().unwrap().len(), 1);
            control
                .abort(RealtimeClose::normal("test cleanup"))
                .await
                .unwrap();
        };
        let (driver_result, ()) = futures::join!(driver.run(), app);
        driver_result.unwrap();
    });
}

#[test]
fn remote_eof_is_success_only_after_session_closed() {
    block_on(async {
        let (session, driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let _ = next_json(&mut peer).await;
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(json_frame(json!({
                "type":"session.closed",
                "reason":"remote_hangup",
                "session":{"id":"live_1"},
                "usage":{"seconds":1.0}
            })))
            .unwrap();
        drop(peer.incoming);
        let (driver_result, event) = futures::join!(driver.run(), events.next());
        driver_result.unwrap();
        assert!(matches!(
            event,
            Some(OpenAiLiveEvent::SessionStarted { .. })
        ));
        // The close event is queued next and remains observable even though the
        // peer closed the transport immediately afterwards.
        assert!(matches!(
            events.next().await,
            Some(OpenAiLiveEvent::SessionClosed { .. })
        ));
        control
            .close_after_session_closed(RealtimeClose::normal("already closed by provider"))
            .await
            .unwrap();
    });
}

#[test]
fn remote_eof_before_session_closed_is_an_interruption() {
    block_on(async {
        let (session, driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let _ = next_json(&mut peer).await;
        let (control, mut events) = session.into_parts();
        drop(peer.incoming);
        let (driver_result, (ready, interrupted)) = futures::join!(driver.run(), async {
            let ready = events.next().await;
            let interrupted = events.next().await;
            (ready, interrupted)
        });
        assert!(matches!(
            driver_result,
            Err(RealtimeError::UnexpectedRemoteClose)
        ));
        assert!(matches!(
            ready,
            Some(OpenAiLiveEvent::SessionStarted { .. })
        ));
        assert!(matches!(
            interrupted,
            Some(OpenAiLiveEvent::ConnectionInterrupted { .. })
        ));
        control
            .abort(RealtimeClose::normal("already interrupted"))
            .await
            .unwrap();
    });
}

#[test]
fn malformed_events_and_provider_errors_remain_observable() {
    block_on(async {
        let (session, driver, _transport, mut peer) =
            connect_with(OpenAiLiveConfig::default(), &[started("gpt-live-1")])
                .await
                .unwrap();
        let _ = next_json(&mut peer).await;
        let (_control, mut events) = session.into_parts();
        let app = async {
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::SessionStarted { .. })
            ));
            peer.incoming
                .unbounded_send(json_frame(json!({
                    "type":"error",
                    "error":{"type":"invalid_request_error","code":"bad_event","message":"Rejected","client_event_id":"cmd_7"}
                })))
                .unwrap();
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::ProviderError { code: Some(code), message, client_event_id: Some(id), .. }) if code == "bad_event" && message == "Rejected" && id == "cmd_7"
            ));
            peer.incoming
                .unbounded_send(json_frame(
                    json!({"type":"session.output_audio.delta","delta":"not-base64!"}),
                ))
                .unwrap();
            assert!(matches!(
                events.next().await,
                Some(OpenAiLiveEvent::ConnectionInterrupted { .. })
            ));
        };
        let (driver_result, ()) = futures::join!(driver.run(), app);
        assert!(matches!(driver_result, Err(RealtimeError::Codec { .. })));
    });
}
