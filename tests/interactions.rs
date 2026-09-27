use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{interactions::*, protocol::*, *};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Mock {
    replies: Mutex<VecDeque<(u16, Value)>>,
    sent: Mutex<Vec<HttpRequest>>,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let (status, value) = self.replies.lock().unwrap().pop_front().unwrap();
        let bytes = match &value {
            Value::String(text) if text.starts_with("data:") => text.as_bytes().to_vec(),
            _ => serde_json::to_vec(&value).unwrap(),
        };
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "req-gemini".into())],
            body: stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }
}
fn setup(replies: Vec<(u16, Value)>) -> (LlmClient, Arc<Mock>) {
    let mut profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"google", "profile_name":"gemini", "base_url":"https://generativelanguage.googleapis.com/v1beta",
        "protocol":"gemini_generate_content", "auth":"api_key", "models":[],
        "interactions":{"mode":"enabled","value":{"endpoint":"https://generativelanguage.googleapis.com/v1beta/interactions","auth":{"type":"api_key","header":"x-goog-api-key"}}}
    })).unwrap();
    profile.regions = vec![Region::International];
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        sent: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}
fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("key".into())),
        account_scope: Some("account-1".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn model_create_continue_and_get_preserve_native_steps() {
    let (client, mock) = setup(vec![
        (
            200,
            json!({"id":"v1_one","model":"gemini-3.8-flash","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"Hello"}]}]}),
        ),
        (
            200,
            json!({"id":"v1_two","model":"gemini-3.8-flash","status":"completed","steps":[]}),
        ),
        (
            200,
            json!({"id":"v1_two","model":"gemini-3.8-flash","status":"completed","steps":[{"type":"user_input"}]}),
        ),
    ]);
    let first = client
        .interactions()
        .create(
            "gemini",
            &InteractionRequest::model("gemini-3.8-flash", "Hi"),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(first.native["steps"][0]["content"][0]["text"], "Hello");
    assert_eq!(first.request_id.as_deref(), Some("req-gemini"));
    let mut second_req = InteractionRequest::model("gemini-3.8-flash", "More");
    second_req.previous = first.reference;
    let second = client
        .interactions()
        .create("gemini", &second_req, &options())
        .await
        .unwrap();
    let fetched = client
        .interactions()
        .get(second.reference.as_ref().unwrap(), &options())
        .await
        .unwrap();
    assert_eq!(fetched.native["steps"][0]["type"], "user_input");
    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent[0].headers[0], ("x-goog-api-key".into(), "key".into()));
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[0].body).unwrap()["model"],
        "gemini-3.8-flash"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[1].body).unwrap()["previous_interaction_id"],
        "v1_one"
    );
    assert_eq!(sent[2].method, "GET");
    assert!(sent[2].url.ends_with("/interactions/v1_two"));
}

#[tokio::test]
async fn agent_background_and_account_scope_are_enforced() {
    let (client, mock) = setup(vec![(
        200,
        json!({"id":"v1_research","agent":"deep-research-preview-04-2026","status":"in_progress","steps":[]}),
    )]);
    let job = client
        .interactions()
        .create(
            "gemini",
            &InteractionRequest::agent("deep-research-preview-04-2026", "Investigate"),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(job.status, "in_progress");
    {
        let sent = mock.sent.lock().unwrap();
        let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
        assert_eq!(body["agent"], "deep-research-preview-04-2026");
        assert_eq!(body["background"], true);
    }
    let mut other = options();
    other.account_scope = Some("account-2".into());
    assert!(matches!(
        client
            .interactions()
            .get(job.reference.as_ref().unwrap(), &other)
            .await,
        Err(InteractionError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(matches!(
        client
            .interactions()
            .delete(job.reference.as_ref().unwrap(), &other)
            .await,
        Err(InteractionError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn provider_error_and_stateless_result_are_explicit() {
    let (client, _) = setup(vec![
        (429, json!({"error":{"message":"quota"}})),
        (
            200,
            json!({"id":"v1_ephemeral","status":"completed","steps":[]}),
        ),
    ]);
    assert!(matches!(
        client
            .interactions()
            .create(
                "gemini",
                &InteractionRequest::model("gemini-2.5-flash", "Hi"),
                &options()
            )
            .await,
        Err(InteractionError::Provider { status: 429, .. })
    ));
    let mut request = InteractionRequest::model("gemini-2.5-flash", "Hi");
    request.store = false;
    let result = client
        .interactions()
        .create("gemini", &request, &options())
        .await
        .unwrap();
    assert!(result.reference.is_none());
}

#[tokio::test]
async fn stream_preserves_native_steps_and_durable_reference() {
    let events = "data: {\"event_type\":\"interaction.created\",\"event_id\":\"e1\",\"interaction\":{\"id\":\"v1_stream\",\"status\":\"in_progress\"}}\n\ndata: {\"event_type\":\"step.delta\",\"event_id\":\"e2\",\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"Hi\"}}\n\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"v1_stream\",\"status\":\"completed\"}}\n\ndata: [DONE]\n\n";
    let (client, mock) = setup(vec![(200, Value::String(events.into()))]);
    let mut stream = client
        .interactions()
        .create_stream(
            "gemini",
            &InteractionRequest::model("gemini-2.5-flash", "Hi"),
            &options(),
        )
        .await
        .unwrap();
    let created = stream.next_event().await.unwrap().unwrap();
    assert_eq!(created.event_type, "interaction.created");
    assert_eq!(created.event_id.as_deref(), Some("e1"));
    assert_eq!(created.reference.unwrap().id, "v1_stream");
    assert_eq!(
        stream.next_event().await.unwrap().unwrap().native["delta"]["text"],
        "Hi"
    );
    assert_eq!(
        stream.next_event().await.unwrap().unwrap().event_type,
        "interaction.completed"
    );
    assert!(stream.next_event().await.unwrap().is_none());
    let sent = mock.sent.lock().unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[0].body).unwrap()["stream"],
        true
    );
}

#[tokio::test]
async fn stream_without_done_reports_bound_reference() {
    let events = "data: {\"event_type\":\"interaction.created\",\"interaction\":{\"id\":\"v1_partial\"}}\n\n";
    let (client, _) = setup(vec![(200, Value::String(events.into()))]);
    let mut stream = client
        .interactions()
        .create_stream(
            "gemini",
            &InteractionRequest::model("gemini-2.5-flash", "Hi"),
            &options(),
        )
        .await
        .unwrap();
    stream.next_event().await.unwrap();
    assert!(matches!(
        stream.next_event().await,
        Err(InteractionStreamError::Interrupted {
            reference: Some(_),
            ..
        })
    ));
}

#[tokio::test]
async fn stream_rejects_changing_id_and_preserves_http_error() {
    let events = "data: {\"event_type\":\"interaction.created\",\"interaction\":{\"id\":\"v1_first\"}}\n\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"v1_other\"}}\n\ndata: [DONE]\n\n";
    let (client, _) = setup(vec![
        (200, Value::String(events.into())),
        (429, json!({"error":{"message":"quota"}})),
    ]);
    let mut stream = client
        .interactions()
        .create_stream(
            "gemini",
            &InteractionRequest::model("gemini-2.5-flash", "Hi"),
            &options(),
        )
        .await
        .unwrap();
    stream.next_event().await.unwrap();
    assert!(matches!(
        stream.next_event().await,
        Err(InteractionStreamError::InvalidEvent {
            reference: Some(_),
            ..
        })
    ));
    assert!(matches!(
        client
            .interactions()
            .create_stream(
                "gemini",
                &InteractionRequest::model("gemini-2.5-flash", "Hi"),
                &options()
            )
            .await,
        Err(InteractionError::Provider { status: 429, .. })
    ));
}

#[tokio::test]
async fn interrupted_stream_resumes_after_the_last_opaque_event_cursor() {
    let interrupted = "data: {\"event_type\":\"interaction.created\",\"event_id\":\"evt/1 +?\",\"interaction\":{\"id\":\"v1_resume\",\"status\":\"in_progress\"}}\n\n";
    let resumed = "data: {\"event_type\":\"interaction.completed\",\"event_id\":\"evt-2\",\"interaction\":{\"id\":\"v1_resume\",\"status\":\"completed\"}}\n\ndata: [DONE]\n\n";
    let (client, mock) = setup(vec![
        (200, Value::String(interrupted.into())),
        (200, Value::String(resumed.into())),
    ]);
    let mut stream = client
        .interactions()
        .create_stream(
            "gemini",
            &InteractionRequest::model("gemini-2.5-flash", "Hi"),
            &options(),
        )
        .await
        .unwrap();
    stream.next_event().await.unwrap().unwrap();
    let reference = stream.reference().unwrap().clone();
    let cursor = stream.last_event_id().unwrap().to_owned();
    assert_eq!(cursor, "evt/1 +?");
    assert!(matches!(
        stream.next_event().await,
        Err(InteractionStreamError::Interrupted { .. })
    ));

    let mut resumed = client
        .interactions()
        .resume_stream(&reference, &cursor, &options())
        .await
        .unwrap();
    assert_eq!(resumed.last_event_id(), Some(cursor.as_str()));
    let event = resumed.next_event().await.unwrap().unwrap();
    assert_eq!(event.event_type, "interaction.completed");
    assert_eq!(event.reference.as_ref().unwrap().id, "v1_resume");
    assert_eq!(resumed.last_event_id(), Some("evt-2"));
    assert!(resumed.next_event().await.unwrap().is_none());

    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent[1].method, "GET");
    let url = url::Url::parse(&sent[1].url).unwrap();
    let query: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
    assert_eq!(query.get("stream").map(String::as_str), Some("true"));
    assert_eq!(
        query.get("last_event_id").map(String::as_str),
        Some("evt/1 +?")
    );
    assert!(url.path().ends_with("/interactions/v1_resume"));
}

#[tokio::test]
async fn background_interaction_can_be_cancelled_then_deleted() {
    let (client, mock) = setup(vec![
        (
            200,
            json!({"id":"v1_background","agent":"deep-research-preview-04-2026","status":"in_progress","steps":[]}),
        ),
        (
            200,
            json!({"id":"v1_background","agent":"deep-research-preview-04-2026","status":"cancelled","steps":[]}),
        ),
        (204, Value::Null),
    ]);
    let job = client
        .interactions()
        .create(
            "gemini",
            &InteractionRequest::agent("deep-research-preview-04-2026", "Research"),
            &options(),
        )
        .await
        .unwrap();
    let reference = job.reference.unwrap();
    let cancelled = client
        .interactions()
        .cancel(&reference, &options())
        .await
        .unwrap();
    assert_eq!(cancelled.status, "cancelled");
    client
        .interactions()
        .delete(&reference, &options())
        .await
        .unwrap();

    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent[1].method, "POST");
    assert!(sent[1].url.ends_with("/interactions/v1_background/cancel"));
    assert_eq!(sent[2].method, "DELETE");
    assert!(sent[2].url.ends_with("/interactions/v1_background"));
}

#[tokio::test]
async fn multimodal_content_uses_interactions_content_blocks() {
    let (client, mock) = setup(vec![(
        200,
        json!({"id":"v1_multimodal","status":"completed","steps":[]}),
    )]);
    let request = InteractionRequest::model(
        "gemini-3.8-flash",
        InteractionInput::content([
            InteractionContent::text("Describe these files."),
            InteractionContent::image_data("aW1hZ2U=", "image/png"),
            InteractionContent::audio_uri(
                "https://generativelanguage.googleapis.com/files/audio-1",
                Some("audio/mp3".into()),
            ),
            InteractionContent::document_uri(
                "https://example.test/report.pdf",
                Some("application/pdf".into()),
            ),
            InteractionContent::video_uri("https://example.test/clip.mp4"),
        ]),
    );

    client
        .interactions()
        .create("gemini", &request, &options())
        .await
        .unwrap();

    let sent = mock.sent.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(
        body["input"][0],
        json!({"type":"text","text":"Describe these files."})
    );
    assert_eq!(
        body["input"][1],
        json!({"type":"image","data":"aW1hZ2U=","mime_type":"image/png"})
    );
    assert_eq!(
        body["input"][2],
        json!({"type":"audio","uri":"https://generativelanguage.googleapis.com/files/audio-1","mime_type":"audio/mp3"})
    );
    assert_eq!(
        body["input"][3],
        json!({"type":"document","uri":"https://example.test/report.pdf","mime_type":"application/pdf"})
    );
    assert_eq!(
        body["input"][4],
        json!({"type":"video","uri":"https://example.test/clip.mp4"})
    );
}

#[tokio::test]
async fn function_tools_and_result_steps_use_native_interactions_shape() {
    let (client, mock) = setup(vec![
        (
            200,
            json!({"id":"v1_tool","status":"requires_action","steps":[{"type":"function_call","id":"call_weather","name":"get_weather","arguments":{"city":"Paris"}}]}),
        ),
        (
            200,
            json!({"id":"v1_after_tool","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"Sunny"}]}]}),
        ),
    ]);
    let tool = InteractionTool::function(
        "get_weather",
        "Gets the weather for a city.",
        json!({
            "type":"object",
            "properties":{"city":{"type":"string"}},
            "required":["city"]
        }),
    );
    let first = client
        .interactions()
        .create(
            "gemini",
            &InteractionRequest::model("gemini-3.8-flash", "Weather in Paris?")
                .with_tool(tool.clone())
                .with_generation_config(json!({
                    "tool_choice":{"allowed_tools":{"mode":"any","tools":["get_weather"]}}
                })),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(first.native["steps"][0]["type"], "function_call");

    let mut next = InteractionRequest::model(
        "gemini-3.8-flash",
        InteractionInput::steps([json!({
            "type":"function_result",
            "name":"get_weather",
            "call_id":"call_weather",
            "result":[{"type":"text","text":"{\"weather\":\"sunny\"}"}]
        })]),
    )
    .with_tool(tool);
    next.previous = first.reference;
    client
        .interactions()
        .create("gemini", &next, &options())
        .await
        .unwrap();

    let sent = mock.sent.lock().unwrap();
    let first_body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(
        first_body["tools"][0],
        json!({
            "type":"function",
            "name":"get_weather",
            "description":"Gets the weather for a city.",
            "parameters":{
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
            }
        })
    );
    assert_eq!(
        first_body["generation_config"]["tool_choice"]["allowed_tools"]["mode"],
        "any"
    );
    let second_body: Value = serde_json::from_slice(&sent[1].body).unwrap();
    assert_eq!(second_body["previous_interaction_id"], "v1_tool");
    assert_eq!(second_body["input"][0]["type"], "function_result");
    assert_eq!(second_body["input"][0]["call_id"], "call_weather");
}

#[tokio::test]
async fn gemini_3_remote_mcp_is_rejected_before_dispatch() {
    let (client, mock) = setup(vec![]);
    let tool = InteractionTool::from_value(json!({
        "type": "mcp_server",
        "name": "weather",
        "url": "https://example.test/mcp"
    }))
    .unwrap();
    let request = InteractionRequest::model("gemini-3.8-flash", "Weather?").with_tool(tool);
    let error = client
        .interactions()
        .create("gemini", &request, &options())
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("Gemini 3 Interactions do not support remote MCP"));
    assert!(mock.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn streaming_create_uses_the_multimodal_and_tool_encoder() {
    let (client, mock) = setup(vec![(200, Value::String(String::new()))]);
    let request = InteractionRequest::model(
        "gemini-3.8-flash",
        InteractionInput::content([
            InteractionContent::text("Describe this image."),
            InteractionContent::image_uri(
                "https://example.test/photo.png",
                Some("image/png".into()),
            ),
        ]),
    )
    .with_tool(InteractionTool::google_search());
    let _stream = client
        .interactions()
        .create_stream("gemini", &request, &options())
        .await
        .unwrap();
    let sent = mock.sent.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["input"][1]["uri"], "https://example.test/photo.png");
    assert_eq!(body["tools"][0], json!({"type":"google_search"}));
}
