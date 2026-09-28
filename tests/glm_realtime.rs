use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::mpsc,
    executor::block_on,
    stream::{self, BoxStream},
    StreamExt,
};
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeConfig;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeEvent;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeFunctionTool;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeOutputAudioFormat;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeRoute;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeScope;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeSession;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeToolChoice;
use lingxi_llm_client::providers::zhipu::realtime::GlmRealtimeTurnDetection;
use lingxi_llm_client::providers::zhipu::realtime::GLM_REALTIME_MAINLAND_ENDPOINT;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{
        RealtimeAudioFormat, RealtimeClose, RealtimeConnectRequest, RealtimeConnection,
        RealtimeError, RealtimeEvent, RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSink,
        RealtimeToolResult, RealtimeTransport,
    },
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

type Incoming = Result<RealtimeFrame, RealtimeError>;

struct FakeTransport {
    incoming: Mutex<Option<BoxStream<'static, Incoming>>>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
}

struct FakePeer {
    incoming: mpsc::UnboundedSender<Incoming>,
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
    request: Arc<Mutex<Option<RealtimeConnectRequest>>>,
    connect_count: Arc<Mutex<usize>>,
}

struct FakeSink {
    sent: Arc<Mutex<Vec<RealtimeFrame>>>,
}

fn fake_transport(preface: &[Value]) -> (Arc<FakeTransport>, FakePeer) {
    let (incoming_tx, incoming_rx) = mpsc::unbounded();
    let prefix = preface
        .iter()
        .map(|value| Ok(RealtimeFrame::text(value.to_string())))
        .collect::<Vec<_>>();
    let incoming = stream::iter(prefix).chain(incoming_rx).boxed();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let request = Arc::new(Mutex::new(None));
    let connect_count = Arc::new(Mutex::new(0));
    (
        Arc::new(FakeTransport {
            incoming: Mutex::new(Some(incoming)),
            sent: sent.clone(),
            request: request.clone(),
            connect_count: connect_count.clone(),
        }),
        FakePeer {
            incoming: incoming_tx,
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
            .expect("one connection per fake transport");
        Ok(RealtimeConnection {
            outbound: Box::new(FakeSink {
                sent: self.sent.clone(),
            }),
            inbound: incoming,
        })
    }
}

#[async_trait]
impl RealtimeSink for FakeSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.sent.lock().unwrap().push(frame);
        Ok(())
    }

    async fn close(&mut self, _close: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}

fn route() -> GlmRealtimeRoute {
    GlmRealtimeRoute::mainland_china()
}

fn scope(route: &GlmRealtimeRoute) -> GlmRealtimeScope {
    GlmRealtimeScope::new("glm-prod", "zhipu-account-7", route).unwrap()
}

fn limits() -> RealtimeLimits {
    RealtimeLimits {
        outbound_capacity: 8,
        event_capacity: 16,
        max_frame_bytes: 16 * 1024,
    }
}

fn setup_events() -> [Value; 3] {
    [
        json!({"type":"session.created","session":{"id":"sess-1"}}),
        json!({"type":"conversation.created","conversation":{"id":"conv-1"}}),
        json!({"type":"session.updated","session":{"input_audio_format":"wav"}}),
    ]
}

fn sent_json(sent: &Arc<Mutex<Vec<RealtimeFrame>>>) -> Vec<Value> {
    sent.lock()
        .unwrap()
        .iter()
        .map(|frame| match frame {
            RealtimeFrame::Text(bytes) => serde_json::from_slice(bytes).unwrap(),
            RealtimeFrame::Binary(_) => panic!("GLM's documented events use JSON text frames"),
        })
        .collect()
}

fn transport(transport: Arc<FakeTransport>) -> Arc<dyn RealtimeTransport> {
    transport
}

#[test]
fn setup_binds_mainland_route_account_and_ephemeral_bearer_credential() {
    block_on(async {
        let route = route();
        let scope = scope(&route);
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route,
            scope,
            Secret::from("per-call-token".to_owned()),
            GlmRealtimeConfig {
                instructions: Some("Keep replies concise.".into()),
                ..GlmRealtimeConfig::default()
            },
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            assert_eq!(
                events.next().await,
                Some(GlmRealtimeEvent::SessionCreated {
                    session: json!({"id":"sess-1"}),
                    native: setup_events()[0].clone(),
                })
            );
            assert_eq!(
                events.next().await,
                Some(GlmRealtimeEvent::ConversationCreated {
                    conversation: json!({"id":"conv-1"}),
                    native: setup_events()[1].clone(),
                })
            );
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::SessionUpdated { .. })
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let request = peer.request.lock().unwrap();
        let request = request.as_ref().unwrap();
        assert_eq!(request.endpoint, GLM_REALTIME_MAINLAND_ENDPOINT);
        assert_eq!(
            request.headers,
            [("Authorization".into(), "Bearer per-call-token".into())]
        );
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
        assert!(!format!("{request:?}").contains("per-call-token"));

        let sent = sent_json(&peer.sent);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0]["type"], "session.update");
        assert_eq!(sent[0]["session"]["input_audio_format"], "wav");
        assert_eq!(sent[0]["session"]["output_audio_format"], "pcm");
        assert_eq!(sent[0]["session"]["instructions"], "Keep replies concise.");
        assert_eq!(sent[0]["session"]["turn_detection"]["type"], "server_vad");
        assert_eq!(sent[0]["session"]["beta_fields"]["chat_mode"], "audio");
        assert_eq!(sent[0]["session"]["beta_fields"]["tts_source"], "e2e");
        assert_eq!(sent[0]["session"]["beta_fields"]["auto_search"], false);
    });
}

#[test]
fn clear_audio_uses_glm_native_input_buffer_event() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            control.send(RealtimeInput::ClearAudio).unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[1], json!({"type":"input_audio_buffer.clear"}));
    });
}

#[test]
fn client_vad_audio_commit_and_text_use_glm_native_events() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let config = GlmRealtimeConfig {
            turn_detection: GlmRealtimeTurnDetection::ClientVad,
            output_audio_format: GlmRealtimeOutputAudioFormat::Mp3,
            ..Default::default()
        };
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            config,
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();

        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            control
                .send(RealtimeInput::Audio {
                    data: Bytes::from_static(b"wav-fragment"),
                    format: RealtimeAudioFormat::Encoded {
                        mime_type: "audio/wav".into(),
                    },
                })
                .unwrap();
            control.send(RealtimeInput::CommitAudio).unwrap();
            control.send(RealtimeInput::ContinueResponse).unwrap();
            control.send(RealtimeInput::Text("Hello".into())).unwrap();
            control.interrupt().unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(sent[0]["session"]["output_audio_format"], "mp3");
        assert_eq!(
            sent[1],
            json!({"type":"input_audio_buffer.append","audio":"d2F2LWZyYWdtZW50"})
        );
        assert_eq!(sent[2], json!({"type":"input_audio_buffer.commit"}));
        assert_eq!(sent[3], json!({"type":"response.create"}));
        assert_eq!(sent[4]["type"], "conversation.item.create");
        assert_eq!(sent[4]["item"]["content"][0]["type"], "input_text");
        assert_eq!(sent[4]["item"]["content"][0]["text"], "Hello");
        assert_eq!(sent[5], json!({"type":"response.create"}));
        assert_eq!(sent[6], json!({"type":"response.cancel"}));
    });
}

#[test]
fn server_vad_rejects_audio_commit_but_allows_explicit_tool_continuation_and_cancel() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                control.send(RealtimeInput::CommitAudio),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.send(RealtimeInput::ContinueResponse).unwrap();
            control.interrupt().unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        assert_eq!(
            &sent_json(&peer.sent)[1..],
            [
                json!({"type":"response.create"}),
                json!({"type":"response.cancel"})
            ]
        );
    });
}

#[test]
fn audio_transcript_output_usage_and_provider_errors_are_typed_and_preserved() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                json!({"type":"response.audio.delta","response_id":"resp-1","item_id":"item-1","output_index":0,"content_index":0,"delta":"AQID"}).to_string(),
            )))
            .unwrap();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                json!({"type":"response.audio_transcript.delta","response_id":"resp-1","delta":"hello"}).to_string(),
            )))
            .unwrap();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                json!({"type":"response.audio_transcript.done","response_id":"resp-1","item_id":"item-1","transcript":"hello there"}).to_string(),
            )))
            .unwrap();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                json!({"type":"response.done","response":{"id":"resp-1","status":"completed","usage":{"input_tokens":3,"output_tokens":5}}}).to_string(),
            )))
            .unwrap();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                json!({"type":"error","error":{"type":"invalid_request_error","code":"bad_event","message":"unsupported field"}}).to_string(),
            )))
            .unwrap();

        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::AudioDelta {
                    response_id: Some(ref id),
                    data,
                    format: GlmRealtimeOutputAudioFormat::Pcm,
                    ..
                }) if id == "resp-1" && data == Bytes::from_static(&[1, 2, 3])
            ));
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::TextDelta { ref delta, .. }) if delta == "hello"
            ));
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::TextDone { ref transcript, .. }) if transcript == "hello there"
            ));
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::ResponseDone { response, .. })
                    if response["usage"]["input_tokens"] == 3
            ));
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::ProviderError {
                    code: Some(ref code),
                    ref message,
                    ..
                }) if code == "bad_event" && message == "unsupported field"
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
    });
}

#[test]
fn scope_mismatch_and_invalid_audio_fail_before_queueing() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let mut wrong_scope: Value = serde_json::to_value(scope(&route)).unwrap();
        wrong_scope["endpoint_fingerprint"] = json!("other-endpoint");
        let wrong_scope: GlmRealtimeScope = serde_json::from_value(wrong_scope).unwrap();
        let result = GlmRealtimeSession::connect(
            transport(fake.clone()),
            route.clone(),
            wrong_scope,
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);

        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                control.send(RealtimeInput::Audio {
                    data: Bytes::from_static(&[1]),
                    format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: 16_000
                    },
                }),
                Err(RealtimeError::InvalidInput { .. })
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        assert_eq!(sent_json(&peer.sent).len(), 1);
    });
}

#[test]
fn session_update_advertises_glm_flattened_function_tools_and_choice() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let config = GlmRealtimeConfig {
            turn_detection: GlmRealtimeTurnDetection::ServerVad,
            tools: vec![GlmRealtimeFunctionTool::new(
                "lookup_order",
                json!({
                    "type":"object",
                    "properties":{"order_number":{"type":"string"}},
                    "required":["order_number"]
                }),
            )
            .with_description("Look up an order.")],
            tool_choice: Some(GlmRealtimeToolChoice::Function("lookup_order".into())),
            ..Default::default()
        };
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            config,
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let setup = sent_json(&peer.sent).remove(0);
        assert_eq!(setup["session"]["tools"][0]["type"], "function");
        assert_eq!(setup["session"]["tools"][0]["name"], "lookup_order");
        assert_eq!(
            setup["session"]["tools"][0]["description"],
            "Look up an order."
        );
        assert_eq!(setup["session"]["tools"][0]["parameters"]["type"], "object");
        assert_eq!(
            setup["session"]["tool_choice"],
            json!({"type":"function","function":"lookup_order"})
        );
        assert_eq!(setup["session"]["turn_detection"]["type"], "server_vad");
    });
}

#[test]
fn function_call_events_keep_optional_call_id_and_native_payload() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let with_call_id = json!({
            "type":"response.function_call_arguments.done",
            "response_id":"resp-1",
            "item_id":"item-1",
            "output_index":0,
            "call_id":"call-1",
            "name":"lookup_order",
            "arguments":"{\"order_number\":\"A-1\"}"
        });
        let without_call_id = json!({
            "type":"response.function_call_arguments.done",
            "response_id":"resp-2",
            "output_index":1,
            "name":"lookup_order",
            "arguments":"{}"
        });
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(with_call_id.to_string())))
            .unwrap();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(without_call_id.to_string())))
            .unwrap();

        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::FunctionCallArgumentsDone {
                    response_id: Some(ref response_id),
                    item_id: Some(ref item_id),
                    output_index: Some(0),
                    call_id: Some(ref call_id),
                    name: Some(ref name),
                    arguments: Some(ref arguments),
                    native,
                }) if response_id == "resp-1"
                    && item_id == "item-1"
                    && call_id == "call-1"
                    && name == "lookup_order"
                    && arguments == "{\"order_number\":\"A-1\"}"
                    && native == with_call_id
            ));
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::FunctionCallArgumentsDone {
                    response_id: Some(ref response_id),
                    output_index: Some(1),
                    call_id: None,
                    native,
                    ..
                }) if response_id == "resp-2" && native == without_call_id
            ));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
    });
}

#[test]
fn tool_outputs_preserve_call_ids_and_continue_only_when_requested() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            control
                .send(RealtimeInput::ToolResult {
                    call_id: "call-single".into(),
                    output: json!({"found":true}),
                })
                .unwrap();
            control
                .send(RealtimeInput::ToolResults {
                    results: vec![
                        RealtimeToolResult {
                            call_id: "call-first".into(),
                            output: json!({"answer":42}),
                        },
                        RealtimeToolResult {
                            call_id: "call-second".into(),
                            output: json!(["ready"]),
                        },
                    ],
                })
                .unwrap();
            control.send(RealtimeInput::ContinueResponse).unwrap();
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();

        let sent = sent_json(&peer.sent);
        assert_eq!(sent.len(), 5);
        assert_eq!(sent[1]["type"], "conversation.item.create");
        assert_eq!(
            sent[1]["item"],
            json!({
                "type":"function_call_output",
                "call_id":"call-single",
                "output":"{\"found\":true}"
            })
        );
        assert_eq!(sent[2]["item"]["call_id"], "call-first");
        assert_eq!(sent[2]["item"]["output"], "{\"answer\":42}");
        assert_eq!(sent[3]["item"]["call_id"], "call-second");
        assert_eq!(sent[3]["item"]["output"], "[\"ready\"]");
        assert_eq!(sent[4], json!({"type":"response.create"}));
        for frame in &sent[1..4] {
            assert!(frame["item"].get("id").is_none());
            assert!(frame.get("response_id").is_none());
        }
    });
}

#[test]
fn invalid_function_result_batch_and_tool_setup_fail_without_partial_send() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let config = GlmRealtimeConfig {
            tools: vec![GlmRealtimeFunctionTool::new(
                "lookup",
                json!("not a JSON Schema object"),
            )],
            ..Default::default()
        };
        let result = GlmRealtimeSession::connect(
            transport(fake.clone()),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            config,
            limits(),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
        assert_eq!(*peer.connect_count.lock().unwrap(), 0);

        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            RealtimeLimits {
                max_frame_bytes: 512,
                ..limits()
            },
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            let error = control
                .send(RealtimeInput::ToolResults {
                    results: vec![
                        RealtimeToolResult {
                            call_id: "call-small".into(),
                            output: json!({"ok":true}),
                        },
                        RealtimeToolResult {
                            call_id: "call-large".into(),
                            output: Value::String("x".repeat(450)),
                        },
                    ],
                })
                .unwrap_err();
            assert!(matches!(error, RealtimeError::FrameTooLarge { .. }));
            control.close(RealtimeClose::normal("done")).await.unwrap();
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        driver_result.unwrap();
        let sent = sent_json(&peer.sent);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0]["type"], "session.update");
    });
}

#[test]
fn unexpected_remote_close_is_reported_without_automatic_reconnect() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming.close_channel();

        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::Realtime(
                    RealtimeEvent::ConnectionInterrupted { .. }
                ))
            ));
            drop(control);
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        assert!(matches!(
            driver_result,
            Err(RealtimeError::UnexpectedRemoteClose)
        ));
        assert_eq!(*peer.connect_count.lock().unwrap(), 1);
    });
}

#[test]
fn malformed_audio_delta_is_a_bounded_protocol_error() {
    block_on(async {
        let route = route();
        let (fake, peer) = fake_transport(&setup_events());
        let (session, driver) = GlmRealtimeSession::connect(
            transport(fake),
            route.clone(),
            scope(&route),
            Secret::from("ephemeral".to_owned()),
            GlmRealtimeConfig::default(),
            limits(),
        )
        .await
        .unwrap();
        let (control, mut events) = session.into_parts();
        peer.incoming
            .unbounded_send(Ok(RealtimeFrame::text(
                r#"{"type":"response.audio.delta","delta":"not base64!"}"#,
            )))
            .unwrap();
        let session_task = async move {
            let _ = events.next().await;
            let _ = events.next().await;
            let _ = events.next().await;
            assert!(matches!(
                events.next().await,
                Some(GlmRealtimeEvent::Realtime(RealtimeEvent::ProviderError {
                    code: Some(ref code),
                    ..
                })) if code == "invalid_provider_frame"
            ));
            drop(control);
        };
        let ((), driver_result) = futures::join!(session_task, driver.run());
        assert!(matches!(driver_result, Err(RealtimeError::Codec { .. })));
    });
}
