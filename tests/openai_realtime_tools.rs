use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on;
use futures::{stream, StreamExt};
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeCodec;
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeConfig;
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeFunctionTool;
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeToolChoice;
use lingxi_llm_client::providers::openai::realtime::OpenAiRealtimeVoice;
use lingxi_llm_client::realtime::{
    RealtimeCodec, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
    RealtimeInput, RealtimeLimits, RealtimeSession, RealtimeSink, RealtimeToolResult,
    RealtimeTransport,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct CountingTransport(AtomicUsize);

#[async_trait]
impl RealtimeTransport for CountingTransport {
    async fn connect(
        &self,
        _request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(RealtimeError::Transport {
            message: "unexpected connection".into(),
        })
    }
}

struct RecordingTransport(Arc<std::sync::Mutex<Vec<RealtimeFrame>>>);

struct RecordingSink(Arc<std::sync::Mutex<Vec<RealtimeFrame>>>);

#[async_trait]
impl RealtimeSink for RecordingSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.0.lock().unwrap().push(frame);
        Ok(())
    }

    async fn close(
        &mut self,
        _close: lingxi_llm_client::realtime::RealtimeClose,
    ) -> Result<(), RealtimeError> {
        Ok(())
    }
}

#[async_trait]
impl RealtimeTransport for RecordingTransport {
    async fn connect(
        &self,
        _request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        Ok(RealtimeConnection {
            outbound: Box::new(RecordingSink(self.0.clone())),
            inbound: stream::pending::<Result<RealtimeFrame, RealtimeError>>().boxed(),
        })
    }
}

fn frame_json(frame: &RealtimeFrame) -> Value {
    let RealtimeFrame::Text(bytes) = frame else {
        panic!("OpenAI Realtime client events must be JSON text frames");
    };
    serde_json::from_slice(bytes).unwrap()
}

#[test]
fn session_update_advertises_documented_function_tools_and_choice() {
    let mut config = OpenAiRealtimeConfig::default();
    config.tools.push(
        OpenAiRealtimeFunctionTool::new(
            "lookup_order",
            json!({
                "type": "object",
                "properties": {"order_number": {"type": "string"}},
                "required": ["order_number"]
            }),
        )
        .with_description("Look up an order by its order number."),
    );
    config.tool_choice = OpenAiRealtimeToolChoice::Function("lookup_order".into());

    let frames = OpenAiRealtimeCodec::new(config).initial_frames().unwrap();
    let setup = frame_json(&frames[0]);
    assert_eq!(setup["type"], "session.update");
    assert_eq!(setup["session"]["tools"][0]["type"], "function");
    assert_eq!(setup["session"]["tools"][0]["name"], "lookup_order");
    assert_eq!(
        setup["session"]["tools"][0]["description"],
        "Look up an order by its order number."
    );
    assert_eq!(setup["session"]["tools"][0]["parameters"]["type"], "object");
    assert_eq!(
        setup["session"]["tool_choice"],
        json!({"type":"function", "name":"lookup_order"})
    );
}

#[test]
fn session_update_encodes_builtin_and_custom_output_voices() {
    // Do not whitelist names in the client: the service may add named voices.
    let built_in = OpenAiRealtimeConfig {
        voice: Some(OpenAiRealtimeVoice::BuiltIn("future_named_voice".into())),
        ..Default::default()
    };
    let built_in_setup =
        frame_json(&OpenAiRealtimeCodec::new(built_in).initial_frames().unwrap()[0]);
    assert_eq!(
        built_in_setup["session"]["audio"]["output"]["voice"],
        "future_named_voice"
    );

    let custom = OpenAiRealtimeConfig {
        voice: Some(OpenAiRealtimeVoice::Custom {
            id: "voice_project_123".into(),
        }),
        ..Default::default()
    };
    let custom_setup = frame_json(&OpenAiRealtimeCodec::new(custom).initial_frames().unwrap()[0]);
    assert_eq!(
        custom_setup["session"]["audio"]["output"]["voice"],
        json!({"id":"voice_project_123"})
    );

    let invalid = OpenAiRealtimeConfig {
        voice: Some(OpenAiRealtimeVoice::Custom { id: "  ".into() }),
        ..Default::default()
    };
    assert!(OpenAiRealtimeCodec::new(invalid).initial_frames().is_err());
}

#[test]
fn image_input_uses_documented_base64_data_url_item() {
    let frames = OpenAiRealtimeCodec::default()
        .encode(&RealtimeInput::Image {
            data: Bytes::from_static(&[1, 2, 3]),
            mime_type: "image/png".into(),
        })
        .unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frame_json(&frames[0]),
        json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AQID"
                }]
            }
        })
    );

    for (data, mime_type) in [
        (Bytes::new(), "image/png"),
        (Bytes::from_static(b"bytes"), "text/plain"),
        (Bytes::from_static(b"bytes"), "image/"),
    ] {
        assert!(OpenAiRealtimeCodec::default()
            .encode(&RealtimeInput::Image {
                data,
                mime_type: mime_type.into(),
            })
            .is_err());
    }
}

#[test]
fn tool_results_are_inserted_together_and_continuation_is_host_controlled() {
    let codec = OpenAiRealtimeCodec::default();
    let frames = codec
        .encode(&RealtimeInput::ToolResults {
            results: vec![
                RealtimeToolResult {
                    call_id: "call-weather".into(),
                    output: json!({"temperature_c": 18}),
                },
                RealtimeToolResult {
                    call_id: "call-calendar".into(),
                    output: json!({"events": []}),
                },
            ],
        })
        .unwrap();

    assert_eq!(frames.len(), 2);
    assert_eq!(frame_json(&frames[0])["type"], "conversation.item.create");
    assert_eq!(frame_json(&frames[1])["type"], "conversation.item.create");
    assert_eq!(
        frame_json(&frames[0])["item"],
        json!({
            "type": "function_call_output",
            "call_id": "call-weather",
            "output": "{\"temperature_c\":18}"
        })
    );
    assert_eq!(frame_json(&frames[1])["item"]["call_id"], "call-calendar");
    assert_eq!(frame_json(&frames[1])["item"]["output"], "{\"events\":[]}");

    let continuation = codec.encode(&RealtimeInput::ContinueResponse).unwrap();
    assert_eq!(continuation.len(), 1);
    assert_eq!(
        frame_json(&continuation[0]),
        json!({"type":"response.create"})
    );
}

#[test]
fn invalid_tool_batches_and_session_declarations_fail_before_encoding() {
    let codec = OpenAiRealtimeCodec::default();
    assert!(codec
        .encode(&RealtimeInput::ToolResults { results: vec![] })
        .is_err());
    assert!(codec
        .encode(&RealtimeInput::ToolResults {
            results: vec![
                RealtimeToolResult {
                    call_id: "duplicate".into(),
                    output: json!(1),
                },
                RealtimeToolResult {
                    call_id: "duplicate".into(),
                    output: json!(2),
                },
            ],
        })
        .is_err());
    assert!(codec
        .encode(&RealtimeInput::ToolResults {
            results: vec![RealtimeToolResult {
                call_id: "   ".into(),
                output: json!(null),
            }],
        })
        .is_err());

    let mut invalid_schema = OpenAiRealtimeConfig::default();
    invalid_schema.tools.push(OpenAiRealtimeFunctionTool::new(
        "lookup",
        json!("not a schema object"),
    ));
    assert!(OpenAiRealtimeCodec::new(invalid_schema)
        .initial_frames()
        .is_err());

    let duplicate_names = OpenAiRealtimeConfig {
        tools: vec![
            OpenAiRealtimeFunctionTool::new("lookup", json!({"type":"object"})),
            OpenAiRealtimeFunctionTool::new("lookup", json!({"type":"object"})),
        ],
        ..Default::default()
    };
    assert!(OpenAiRealtimeCodec::new(duplicate_names)
        .initial_frames()
        .is_err());

    let missing_choice = OpenAiRealtimeConfig {
        tool_choice: OpenAiRealtimeToolChoice::Function("missing".into()),
        ..Default::default()
    };
    assert!(OpenAiRealtimeCodec::new(missing_choice)
        .initial_frames()
        .is_err());
}

#[test]
fn invalid_tool_setup_is_rejected_before_transport_connects() {
    let config = OpenAiRealtimeConfig {
        tool_choice: OpenAiRealtimeToolChoice::Required,
        ..Default::default()
    };
    let transport = CountingTransport(AtomicUsize::new(0));
    let result = block_on(RealtimeSession::connect(
        &transport,
        RealtimeConnectRequest {
            endpoint: "wss://api.openai.com/v1/realtime".into(),
            headers: Vec::new(),
            max_frame_bytes: 1024,
        },
        Arc::new(OpenAiRealtimeCodec::new(config)),
        RealtimeLimits::default(),
    ));

    assert!(matches!(result, Err(RealtimeError::InvalidConfig { .. })));
    assert_eq!(transport.0.load(Ordering::SeqCst), 0);
}

#[test]
fn oversized_second_tool_result_does_not_send_the_first_result() {
    let sent = Arc::new(std::sync::Mutex::new(Vec::new()));
    let transport = RecordingTransport(sent.clone());
    let limits = RealtimeLimits {
        max_frame_bytes: 512,
        ..RealtimeLimits::default()
    };
    let (session, driver) = block_on(RealtimeSession::connect(
        &transport,
        RealtimeConnectRequest {
            endpoint: "wss://api.openai.com/v1/realtime".into(),
            headers: Vec::new(),
            max_frame_bytes: 512,
        },
        Arc::new(OpenAiRealtimeCodec::default()),
        limits,
    ))
    .unwrap();
    let (control, _events) = session.into_parts();
    let error = control
        .send(RealtimeInput::ToolResults {
            results: vec![
                RealtimeToolResult {
                    call_id: "call-small".into(),
                    output: json!({"ok": true}),
                },
                RealtimeToolResult {
                    call_id: "call-large".into(),
                    output: Value::String("x".repeat(450)),
                },
            ],
        })
        .unwrap_err();
    assert!(matches!(error, RealtimeError::FrameTooLarge { .. }));

    drop(control);
    block_on(driver.run()).unwrap();
    let sent = sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(frame_json(&sent[0])["type"], "session.update");
}

#[test]
fn oversized_encoded_image_is_rejected_without_sending_a_frame() {
    let sent = Arc::new(std::sync::Mutex::new(Vec::new()));
    let transport = RecordingTransport(sent.clone());
    let limits = RealtimeLimits {
        max_frame_bytes: 512,
        ..RealtimeLimits::default()
    };
    let (session, driver) = block_on(RealtimeSession::connect(
        &transport,
        RealtimeConnectRequest {
            endpoint: "wss://api.openai.com/v1/realtime".into(),
            headers: Vec::new(),
            max_frame_bytes: 512,
        },
        Arc::new(OpenAiRealtimeCodec::default()),
        limits,
    ))
    .unwrap();
    let (control, _events) = session.into_parts();
    let data = Bytes::from(vec![0_u8; 300]);
    assert!(data.len() + "image/png".len() < limits.max_frame_bytes);
    let error = control
        .send(RealtimeInput::Image {
            data,
            mime_type: "image/png".into(),
        })
        .unwrap_err();
    assert!(matches!(error, RealtimeError::FrameTooLarge { .. }));

    drop(control);
    block_on(driver.run()).unwrap();
    let sent = sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(frame_json(&sent[0])["type"], "session.update");
}
