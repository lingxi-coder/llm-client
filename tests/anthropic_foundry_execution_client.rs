use async_trait::async_trait;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"custom-foundry", "profile_name":"foundry",
        "protocol":"foundry_claude", "base_url":"https://test.services.ai.azure.com/anthropic",
        "auth":"none", "regions":["international"],
        "models":[{"display_model":"enabled", "request_model":"deployment",
            "billing_model":"unrelated", "foundry":{"hosting":"anthropic","model_id":"claude-opus-5-5"}}]
    })).unwrap()
}
fn request(model: &str) -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model":model,"messages":[{"role":"user","content":[{"type":"text","text":"Calculate 1+1"}]}]
    })).unwrap();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
            Default::default(),
        )
        .into(),
    );
    request
}
#[derive(Default)]
struct Recording {
    requests: Mutex<Vec<HttpRequest>>,
    fail: bool,
}
#[async_trait]
impl Transport for Recording {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        self.requests.lock().unwrap().push(request);
        if self.fail {
            return Err(LlmError::Transport {
                message: "unknown execution outcome".into(),
            });
        }
        let usage = json!({"input_tokens":2,"output_tokens":1,"server_tool_use":{"code_execution_requests":1}});
        let container = json!({"id":"container_foundry","future_field":"preserve"});
        let message = json!({"id":"msg1","type":"message","role":"assistant","model":"deployment","content":[],"stop_reason":"end_turn","usage":usage,"container":container});
        let bytes = if body["stream"] == true {
            let events = [
                json!({"type":"message_start","message":message}),
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":usage}),
                json!({"type":"message_stop"}),
            ];
            events
                .iter()
                .map(|event| {
                    format!(
                        "event: {}\ndata: {event}\n\n",
                        event["type"].as_str().unwrap()
                    )
                })
                .collect::<String>()
                .into_bytes()
        } else {
            serde_json::to_vec(&message).unwrap()
        };
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: bytes.into(),
        }
        .into())
    }
}
#[tokio::test]
async fn selected_hosting_is_used_and_complete_stream_preserve_execution_metadata() {
    for reverse in [false, true] {
        let mut profile = profile();
        let mut azure = profile.models[0].clone();
        azure.display_model = "disabled".into();
        azure.foundry.as_mut().unwrap().hosting = FoundryHosting::Azure;
        profile.models.push(azure);
        if reverse {
            profile.models.reverse();
        }
        let transport = Arc::new(Recording::default());
        let client = LlmClientBuilder::with_transport(transport.clone(), &[profile])
            .with_region(Region::International)
            .build()
            .unwrap();
        let options = RequestOptions::default();
        let denied = request("disabled");
        assert!(client.chat().complete(&denied, &options).await.is_err());
        assert!(client.chat().stream(&denied, &options).await.is_err());
        assert!(transport.requests.lock().unwrap().is_empty());
        let allowed = request("enabled");
        let response = client.chat().complete(&allowed, &options).await.unwrap();
        assert_eq!(
            response.anthropic_container().unwrap().envelope["future_field"],
            "preserve"
        );
        assert_eq!(
            response.anthropic_usage().unwrap()["server_tool_use"]["code_execution_requests"],
            1
        );
        let mut stream = client.chat().stream(&allowed, &options).await.unwrap();
        let mut ended = false;
        while let Some(event) = stream.next().await {
            ended |= matches!(event.unwrap(), StreamEvent::End { .. });
        }
        assert!(ended);
        assert_eq!(
            stream.anthropic_container().unwrap().envelope["id"],
            "container_foundry"
        );
        assert_eq!(
            stream.anthropic_usage().unwrap()["server_tool_use"]["code_execution_requests"],
            1
        );
        let sent = transport.requests.lock().unwrap();
        assert_eq!(sent.len(), 2);
        for wire in sent.iter() {
            assert_eq!(
                wire.url,
                "https://test.services.ai.azure.com/anthropic/v1/messages"
            );
            let body: Value = serde_json::from_slice(&wire.body).unwrap();
            assert_eq!(body["model"], "deployment");
            assert!(!body.to_string().contains("unrelated"));
        }
    }
}
#[tokio::test]
async fn uncertain_foundry_execution_does_not_retry_or_fail_over() {
    for stream in [false, true] {
        let mut primary = profile();
        primary.connection.group = Some("executors".into());
        primary.connection.failover.network = true;
        let mut backup = primary.clone();
        backup.profile_name = "backup".into();
        backup.connection.order = 1;
        let transport = Arc::new(Recording {
            fail: true,
            ..Default::default()
        });
        let client = LlmClientBuilder::with_transport(transport.clone(), &[primary, backup])
            .with_region(Region::International)
            .build()
            .unwrap();
        let req = request("enabled");
        if stream {
            assert!(client
                .chat()
                .stream(&req, &Default::default())
                .await
                .is_err());
        } else {
            assert!(client
                .chat()
                .complete(&req, &Default::default())
                .await
                .is_err());
        }
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}
