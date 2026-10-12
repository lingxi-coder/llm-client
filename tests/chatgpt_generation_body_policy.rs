use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    auth::Authenticator,
    protocol::{
        AuthStrategy, ChatRequest, ContentBlock, LlmError, ProviderProfile, Region, Secret,
    },
    providers::OpenAiClient,
    HttpRequest, LlmClient, LlmClientBuilder, RequestMode, RequestOptions, StreamResponse,
    Transport,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

const DISPLAY: &str = "\u{fffd}Ω\u{fffd}";
const EXACT_WIRE: &str = "\\ud800Ω\\udc00";

struct NoopChatGptAuth;

#[async_trait]
impl Authenticator for NoopChatGptAuth {
    async fn apply(
        &self,
        _request: &mut HttpRequest,
        _profile: &ProviderProfile,
        _credential: Option<&lingxi_llm_client::protocol::Secret<String>>,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

#[derive(Debug)]
struct ReintroduceChatGptControls;

impl lingxi_llm_client::client::options::RequestFinalizer for ReintroduceChatGptControls {
    fn finalize(
        &self,
        request: &mut HttpRequest,
        _profile: &ProviderProfile,
    ) -> Result<(), LlmError> {
        let mut body: Value = serde_json::from_slice(&request.body).unwrap();
        body["max_output_tokens"] = json!(512);
        body["temperature"] = json!(0.8);
        body["top_p"] = json!(0.9);
        body["stream"] = json!(false);
        body["store"] = json!(true);
        body.as_object_mut().unwrap().remove("instructions");
        request.body = serde_json::to_vec(&body).unwrap().into();
        Ok(())
    }
}

#[derive(Default)]
struct RecordingTransport {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let stream_request = String::from_utf8_lossy(&request.body).contains("\"stream\":true");
        let is_files = request.url.contains("/files");
        let is_embeddings = request.url.contains("/embeddings");
        self.requests.lock().unwrap().push(request);
        let completed = json!({
            "type":"response.completed",
            "response":{
                "id":"resp-test",
                "model":"wire",
                "status":"completed",
                "output":[],
                "usage":{"input_tokens":1,"output_tokens":0}
            }
        });
        let bytes = if is_files {
            serde_json::to_vec(&json!({"data":[]})).unwrap().into()
        } else if is_embeddings {
            serde_json::to_vec(&json!({"data":[{"index":0,"embedding":[0.5]}]}))
                .unwrap()
                .into()
        } else if stream_request {
            Bytes::from(format!("data: {}\n\n", completed))
        } else {
            serde_json::to_vec(&completed["response"]).unwrap().into()
        };
        Ok(StreamResponse {
            status: 200,
            headers: Vec::new(),
            body: stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai",
        "profile_name":"chatgpt",
        "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_responses",
        "auth":"chat_gpt_o_auth",
        "models":[{
            "display_model":"gpt-test",
            "request_model":"wire",
            "billing_model":"wire"
        }],
        "embeddings":{
            "mode":"enabled",
            "value":{
                "api":"open_ai",
                "endpoint":"https://example.test/v1/embeddings",
                "auth":{"type":"bearer"}
            }
        }
    }))
    .unwrap()
}

fn client(transport: Arc<RecordingTransport>) -> LlmClient {
    let mut builder = LlmClientBuilder::with_transport(transport, &[profile()])
        .with_region(Region::International);
    builder.register_authenticator(AuthStrategy::ChatGptOAuth, Arc::new(NoopChatGptAuth));
    builder.build().unwrap()
}

#[tokio::test]
async fn codex_search_retains_lite_tools_through_generation_auth_and_dispatch() {
    let transport = Arc::new(RecordingTransport::default());
    let mut configured = profile();
    configured.base_url = "https://chatgpt.com/backend-api/codex".into();
    configured.extra["web_search"] = json!("openai_responses");
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[configured])
        .with_region(Region::International);
    builder.register_authenticator(AuthStrategy::ChatGptOAuth, Arc::new(NoopChatGptAuth));
    let client = builder.build().unwrap();
    let mut search = request_with_exact_text();
    search.set_hosted_web_search(Some(Default::default()));
    let mut stream = client
        .chat()
        .stream(&search, &RequestOptions::default())
        .await
        .unwrap();
    while let Some(event) = stream.next().await {
        event.unwrap();
    }
    let requests = transport.requests.lock().unwrap();
    let sent = &requests[0];
    assert_eq!(sent.url, "https://chatgpt.com/backend-api/codex/responses");
    let raw = String::from_utf8_lossy(&sent.body);
    assert!(raw.contains(EXACT_WIRE));
    // serde_json cannot decode lone surrogate values; inspect the SDK's
    // semantic projection by using a display-only request for structure.
    let readable = raw.replace(EXACT_WIRE, DISPLAY);
    let body: Value = serde_json::from_str(&readable).unwrap();
    assert!(body.get("tools").is_none());
    assert_eq!(
        body["input"][0]["tools"][0],
        json!({"type":"web_search","external_web_access":true})
    );
    assert_eq!(body["input"][1]["content"][0]["text"], DISPLAY);
    assert_eq!(body["store"], false);
    assert!(body.get("max_output_tokens").is_none());
    assert!(sent
        .headers
        .iter()
        .any(|(name, value)| name == "x-openai-internal-codex-responses-lite" && value == "true"));
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"gpt-test",
        "messages":[{"role":"user","content":[{"type":"text","text":DISPLAY}]}],
        "max_tokens":128,
        "temperature":0.2,
        "controls":{
            "top_p":0.4,
            "responses":{"store":true,"prompt_cache_key":"retain-me"}
        }
    }))
    .unwrap()
}

fn request_with_exact_text() -> ChatRequest {
    let mut request = request();
    request.messages[0].content[0] = ContentBlock::TextJsUtf16 {
        text: DISPLAY.into(),
        utf16_code_units: vec![0xd800, 0x03a9, 0xdc00],
        thought_signature: None,
        citations: None,
    };
    request
}

fn options_with_lone_units() -> RequestOptions {
    RequestOptions {
        message_text_utf16_overrides: [(
            "/messages/0/content/0/text".into(),
            vec![0xd800, 0x03a9, 0xdc00],
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

fn assert_generation_wire(body: &[u8], exact_text: bool) {
    let raw = String::from_utf8_lossy(body).into_owned();
    let display_json = if exact_text {
        raw.replace("\\ud800", "�").replace("\\udc00", "�")
    } else {
        raw.clone()
    };
    let value: Value = serde_json::from_str(&display_json).unwrap();
    assert!(value.get("max_output_tokens").is_none());
    assert!(value.get("temperature").is_none());
    assert!(value.get("top_p").is_none());
    assert_eq!(value["store"], false);
    assert_eq!(value["stream"], true);
    assert_eq!(value["instructions"], "");
    assert_eq!(value["prompt_cache_key"], "retain-me");
    assert!(!raw.contains("utf16_code_units"));
    assert!(!raw.contains("text_js_utf16"));
    if exact_text {
        assert!(raw.contains(EXACT_WIRE));
    }
}

#[tokio::test]
async fn normal_complete_applies_chatgpt_body_policy_before_exact_utf16_serialization() {
    let transport = Arc::new(RecordingTransport::default());
    let client = client(transport.clone());
    let request = request_with_exact_text();
    let mut options = RequestOptions::default();
    options.finalizer = Some(Arc::new(ReintroduceChatGptControls));
    client
        .chat()
        .complete_in("chatgpt", &request, &options)
        .await
        .unwrap();

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_generation_wire(&requests[0].body, true);
}

#[tokio::test]
async fn normal_stream_applies_chatgpt_body_policy_before_dispatch() {
    let transport = Arc::new(RecordingTransport::default());
    let client = client(transport.clone());
    let request = request_with_exact_text();
    let mut stream = client
        .chat()
        .stream_in("chatgpt", &request, &RequestOptions::default())
        .await
        .unwrap();
    while let Some(batch) = stream.next_batch().await {
        for event in batch.events {
            event.unwrap();
        }
    }

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_generation_wire(&requests[0].body, true);
}

#[tokio::test]
async fn count_mode_does_not_apply_generation_auth_body_policy() {
    let transport = Arc::new(RecordingTransport::default());
    let client = client(transport.clone());
    assert_eq!(
        client
            .count_tokens_exact_in("chatgpt", &request(), &RequestOptions::default())
            .await
            .unwrap(),
        None
    );
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn request_mut_reapplies_policy_and_discards_stale_utf16_mappings() {
    let transport = Arc::new(RecordingTransport::default());
    let client = client(transport);
    let mut draft = client
        .prepare_draft_on(
            "chatgpt",
            &request(),
            &options_with_lone_units(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    draft
        .apply_request_body_auth_policy(AuthStrategy::ChatGptOAuth)
        .unwrap();
    assert!(String::from_utf8_lossy(&draft.request().body).contains(EXACT_WIRE));

    let mut changed = draft.semantic_body_json().unwrap();
    changed["max_output_tokens"] = json!(32);
    changed["temperature"] = json!(0.7);
    changed["top_p"] = json!(0.6);
    changed["store"] = json!(true);
    draft.request_mut().body = serde_json::to_vec(&changed).unwrap().into();

    let call = draft.seal().await.unwrap();
    assert_generation_wire(&call.request().body, false);
    assert!(!String::from_utf8_lossy(&call.request().body).contains("\\ud800"));
}

#[tokio::test]
async fn files_and_embeddings_keep_their_service_bodies_outside_chat_policy() {
    let transport = Arc::new(RecordingTransport::default());
    let client = client(transport.clone());

    client
        .provider::<OpenAiClient>("chatgpt")
        .unwrap()
        .files()
        .list(
            None,
            &RequestOptions {
                account_scope: Some("account".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    client
        .embeddings()
        .embed(
            "chatgpt",
            &lingxi_llm_client::embeddings::EmbeddingRequest {
                model: "embedding-test".into(),
                input: vec!["text".into()],
                dimensions: Some(1),
                task: None,
            },
            &RequestOptions {
                credential: Some(Secret::new("test-token".into())),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert!(request.body.is_empty() || serde_json::from_slice::<Value>(&request.body).is_ok());
        if !request.body.is_empty() {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            assert!(body.get("max_output_tokens").is_none());
            assert!(body.get("temperature").is_none());
            assert!(body.get("top_p").is_none());
            assert!(body.get("store").is_none());
            assert!(body.get("instructions").is_none());
        }
    }
}
