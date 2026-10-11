use async_trait::async_trait;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    audio::AudioRoute,
    protocol::{ProviderProfile, Region, Secret, ToolSpec},
    realtime::*,
    LlmClientBuilder,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Transport {
    requests: Mutex<Vec<RealtimeConnectRequest>>,
    frames: Arc<Mutex<Vec<RealtimeFrame>>>,
    google: bool,
}
struct Sink(Arc<Mutex<Vec<RealtimeFrame>>>);
#[async_trait]
impl RealtimeSink for Sink {
    async fn ping(&mut self, _payload: bytes::Bytes) -> Result<(), RealtimeError> {
        Err(RealtimeError::InvalidInput {
            message: "test transport does not support explicit WebSocket Ping frames".into(),
        })
    }

    fn abort(&mut self) {
        // The test transport releases its local state when dropped.
    }

    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.0.lock().unwrap().push(frame);
        Ok(())
    }
    async fn close(&mut self, _: RealtimeClose) -> Result<(), RealtimeError> {
        Ok(())
    }
}
#[async_trait]
impl RealtimeTransport for Transport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        self.requests.lock().unwrap().push(request);
        let inbound = if self.google {
            stream::once(async { Ok(RealtimeFrame::text(r#"{"setupComplete":{}}"#)) })
                .chain(stream::pending())
                .boxed()
        } else {
            stream::pending().boxed()
        };
        Ok(RealtimeConnection {
            outbound: Box::new(Sink(self.frames.clone())),
            inbound,
        })
    }
}
fn snapshot(provider: &str, base: &str) -> lingxi_llm_client::ClientSnapshot {
    let profile:ProviderProfile=serde_json::from_value(json!({"provider_id":provider,"profile_name":"audio-account","base_url":base,"auth":"api_key","protocol":"open_ai_chat","models":[]})).unwrap();
    LlmClientBuilder::new(&[profile])
        .unwrap()
        .with_region(Region::International)
        .build()
        .unwrap()
        .snapshot()
}
fn tool() -> ToolSpec {
    serde_json::from_value(json!({"name":"lookup","description":"lookup a value","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}})).unwrap()
}
fn history() -> Vec<RealtimeHistoryItem> {
    vec![
        RealtimeHistoryItem::Message {
            item_id: None,
            role: RealtimeRole::User,
            text: "earlier question".into(),
        },
        RealtimeHistoryItem::Message {
            item_id: None,
            role: RealtimeRole::Assistant,
            text: "earlier answer".into(),
        },
    ]
}
fn frames(transport: &Transport) -> Vec<Value> {
    transport
        .frames
        .lock()
        .unwrap()
        .iter()
        .map(|frame| {
            let RealtimeFrame::Text(data) = frame else {
                panic!()
            };
            serde_json::from_slice(data).unwrap()
        })
        .collect()
}

#[tokio::test]
async fn openai_connector_requires_no_chat_model_and_builds_auth_tools_history() {
    let transport = Arc::new(Transport::default());
    let connected = connect_audio_conversation(
        &snapshot("openai", "https://api.openai.com/v1"),
        &AudioRoute::new("audio-account", "account"),
        AudioRealtimeConfig::default(),
        vec![tool()],
        history(),
        Secret::new("private-key".into()),
        transport.clone(),
        RealtimeLimits::default(),
    )
    .await
    .unwrap();
    assert_eq!(connected.model, "gpt-realtime-2.1");
    assert!(connected.capabilities.agent_conversation());
    let requests = transport.requests.lock().unwrap();
    assert_eq!(
        requests[0].endpoint,
        "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
    );
    assert_eq!(requests[0].headers[0].1, "Bearer private-key");
    drop(requests);
    let sent = frames(&transport);
    assert_eq!(sent.len(), 3);
    assert!(sent[0]["session"]["audio"]["input"]["turn_detection"].is_null());
    assert_eq!(sent[0]["session"]["tools"][0]["name"], "lookup");
    assert_eq!(sent[1]["item"]["role"], "user");
    assert_eq!(sent[2]["item"]["role"], "assistant");
    assert!(sent.iter().all(|v| v["type"] != "response.create"));
}

#[tokio::test]
async fn google_connector_seeds_no_response_history_and_builds_native_tool_schema() {
    let transport = Arc::new(Transport {
        google: true,
        ..Default::default()
    });
    let connected = connect_audio_conversation(
        &snapshot("google", "https://generativelanguage.googleapis.com/v1beta"),
        &AudioRoute::new("audio-account", "account"),
        AudioRealtimeConfig::default(),
        vec![tool()],
        history(),
        Secret::new("private-key".into()),
        transport.clone(),
        RealtimeLimits::default(),
    )
    .await
    .unwrap();
    assert_eq!(connected.model, "gemini-3.8-live");
    assert_eq!(
        connected.input_format,
        RealtimeAudioFormat::Pcm16 {
            sample_rate_hz: 16000
        }
    );
    let setup = &frames(&transport)[0]["setup"];
    assert_eq!(
        setup["historyConfig"]["initialHistoryInClientContent"],
        true
    );
    assert_eq!(
        setup["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["type"],
        "object"
    );
    assert!(setup.get("inputAudioTranscription").is_some());
    assert!(setup.get("outputAudioTranscription").is_some());
    let task = async {
        connected
            .control
            .close(RealtimeClose::normal("done"))
            .await
            .unwrap()
    };
    let (_, result) = futures::join!(task, connected.driver.run());
    result.unwrap();
    let sent = frames(&transport);
    assert_eq!(sent[1]["clientContent"]["turnComplete"], true);
    assert_eq!(sent[1]["clientContent"]["turns"][1]["role"], "model");
}

#[tokio::test]
async fn unsupported_routes_models_and_tools_fail_before_connect() {
    for (provider, base, config, mut tools) in [
        (
            "google",
            "https://us-central1-aiplatform.googleapis.com",
            AudioRealtimeConfig::default(),
            vec![],
        ),
        (
            "openai",
            "https://api.openai.com/v1",
            AudioRealtimeConfig {
                model: Some("chat-model".into()),
                ..Default::default()
            },
            vec![],
        ),
        (
            "openai",
            "https://api.openai.com/v1",
            AudioRealtimeConfig::default(),
            vec![tool()],
        ),
    ] {
        if !tools.is_empty() {
            tools[0].defer_loading = true;
        }
        let transport = Arc::new(Transport::default());
        assert!(connect_audio_conversation(
            &snapshot(provider, base),
            &AudioRoute::new("audio-account", "account"),
            config,
            tools,
            history(),
            Secret::new("private-key".into()),
            transport.clone(),
            RealtimeLimits::default()
        )
        .await
        .is_err());
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}
