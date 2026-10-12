use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{
        AuthStrategy, ChatRequest, LlmError, NativeExtension, ProviderProfile, Region, Secret,
        UsageState, WebSearchConfig,
    },
    providers::openai::codex_search::CodexSearchRequest,
    HttpRequest, LlmClient, LlmClientBuilder, RequestMode, RequestOptions, StreamResponse,
    Transport,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

struct SearchTransport {
    requests: Mutex<Vec<HttpRequest>>,
    status: u16,
    response: String,
}
#[async_trait]
impl Transport for SearchTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        Ok(StreamResponse {
            status: self.status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: stream::iter(
                self.response
                    .as_bytes()
                    .chunks(5)
                    .map(Bytes::copy_from_slice)
                    .map(Ok)
                    .collect::<Vec<_>>(),
            )
            .boxed(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"openai-chatgpt","profile_name":"codex","auth":"chat_gpt_o_auth",
        "base_url":"https://chatgpt.com/backend-api/codex","protocol":"open_ai_responses","supports_websockets":true,
        "extra":{"web_search":"openai_responses"},"models":[{"display_model":"m","request_model":"m","billing_model":"m"}]})).unwrap()
}
fn client(transport: Arc<SearchTransport>, profile: ProviderProfile) -> LlmClient {
    let mut builder =
        LlmClientBuilder::with_transport(transport, &[profile]).with_region(Region::International);
    builder.register_authenticator(
        AuthStrategy::ChatGptOAuth,
        Arc::new(lingxi_llm_client::BearerAuthenticator),
    );
    builder.build().unwrap()
}
fn request() -> ChatRequest {
    let mut request = ChatRequest::new("m");
    request.max_tokens = Some(4096);
    request.set_hosted_web_search(Some(WebSearchConfig {
        allowed_domains: vec!["example.com".into()],
        ..Default::default()
    }));
    request.native_options.push(
        NativeExtension::from_typed(CodexSearchRequest {
            id: "search-session".into(),
            query: "武汉 今日天气".into(),
        })
        .unwrap(),
    );
    request
}
fn transport(status: u16, response: Value) -> Arc<SearchTransport> {
    Arc::new(SearchTransport {
        requests: Mutex::default(),
        status,
        response: response.to_string(),
    })
}
fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("synthetic-test-token".into())),
        ..Default::default()
    }
}

#[tokio::test]
async fn standalone_search_keeps_native_wire_shape_auth_and_real_sources() {
    for deferred in [false, true] {
        let http = transport(
            200,
            json!({"encrypted_output":"opaque","output":"天气来源","results":[
                {"type":"text_result","title":"天气","url":"https://example.com/weather","ref_id":"turn0search0","future_field":true},
                {"type":"text_result","title":"Duplicate","url":"https://example.com/weather"},
                {"type":"text_result","title":"Unsafe","url":"javascript:alert(1)"}
            ]}),
        );
        let mut p = profile();
        if deferred {
            p.auth = AuthStrategy::None;
        }
        let client = client(http.clone(), p);
        let mut r = request();
        if deferred {
            r.set_hosted_web_search(Some(WebSearchConfig {
                blocked_domains: vec!["blocked.example".into()],
                ..Default::default()
            }));
        }
        let response = if deferred {
            let mut draft = client
                .prepare_draft_on(
                    "codex",
                    &r,
                    &RequestOptions::default(),
                    RequestMode::Complete,
                )
                .await
                .unwrap();
            draft
                .apply_request_body_auth_policy(AuthStrategy::ChatGptOAuth)
                .unwrap();
            let collected = draft
                .seal()
                .await
                .unwrap()
                .dispatch_once()
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let response = collected.decode().unwrap();
            collected.finish().await;
            response
        } else {
            client
                .chat()
                .complete_in("codex", &r, &options())
                .await
                .unwrap()
        };
        assert_eq!(response.message.text(), "天气来源");
        assert_eq!(response.model, "m");
        assert_eq!(response.usage.state, UsageState::Missing);
        let search = response.web_search.unwrap();
        assert_eq!(search.citations.len(), 1);
        assert_eq!(search.citations[0].url, "https://example.com/weather");
        assert_eq!(search.metadata["results"][0]["future_field"], true);
        let requests = http.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let wire = &requests[0];
        assert_eq!(
            wire.url,
            "https://chatgpt.com/backend-api/codex/alpha/search"
        );
        let body: Value = serde_json::from_slice(&wire.body).unwrap();
        assert_eq!(
            body,
            json!({"id":"search-session","model":"m","commands":{"search_query":[{"q":"武汉 今日天气"}]},
            "settings":{"external_web_access":true,"allowed_callers":["direct"],"filters":if deferred {json!({"blocked_domains":["blocked.example"]})} else {json!({"allowed_domains":["example.com"]})}},"max_output_tokens":4096})
        );
        if !deferred {
            assert!(wire
                .headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                    && value == "Bearer synthetic-test-token"));
        }
    }
}

#[tokio::test]
async fn standalone_rejects_streams_other_endpoints_and_invalid_queries_before_dispatch() {
    for case in 0..5 {
        let http = transport(200, json!({"output":"unused"}));
        let mut p = profile();
        let mut r = request();
        if case == 0 {
            p.base_url = "https://example.com/v1".into();
        }
        if case == 1 {
            r.native_options = vec![NativeExtension::from_typed(CodexSearchRequest {
                id: "id".into(),
                query: " ".into(),
            })
            .unwrap()];
        }
        if case == 3 {
            p.provider_id = "custom".into();
            p.auth = AuthStrategy::None;
            p.protocol = lingxi_llm_client::protocol::ProtocolFamily::OpenAiChat;
            p.base_url = "https://example.com/v1".into();
            p.extra["web_search"] = json!("openai_chat");
        }
        if case == 4 {
            r.tool_choice = lingxi_llm_client::protocol::ToolChoice::None;
        }
        let mode = if case == 2 {
            RequestMode::Stream
        } else {
            RequestMode::Complete
        };
        assert!(client(http.clone(), p)
            .prepare_on("codex", &r, &options(), mode)
            .await
            .is_err());
        assert!(http.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn standalone_errors_and_malformed_results_do_not_become_success() {
    for (status, body) in [
        (400, json!({"error":{"message":"search rejected"}})),
        (200, json!({"output":"x","results":{}})),
    ] {
        let http = transport(status, body);
        assert!(client(http, profile())
            .chat()
            .complete_in("codex", &request(), &options())
            .await
            .is_err());
    }
}
