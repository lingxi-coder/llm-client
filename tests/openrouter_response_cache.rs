use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, *};
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Mock {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let body = serde_json::to_vec(&json!({
            "id":"chatcmpl-test", "model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
        }))
        .unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(body.into()) }).boxed(),
        })
    }
}

fn profile(provider: &str, base_url: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":provider, "profile_name":"test-profile", "base_url":base_url,
        "protocol":"open_ai_chat", "auth":"none",
        "models":[{"request_model":"test-model","display_model":"test-model","billing_model":"test-model"}]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"test-model",
        "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
    }))
    .unwrap()
}

#[derive(Default)]
struct ResponsesRouteMock {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for ResponsesRouteMock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let body = serde_json::to_vec(&json!({
            "id":"resp-cache-test",
            "status":"completed",
            "model":"test-model",
            "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
            "usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}
        }))
        .unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![
                ("X-OpenRouter-Cache-Status".into(), "HIT".into()),
                ("X-OpenRouter-Cache-Age".into(), "8".into()),
                ("X-OpenRouter-Cache-TTL".into(), "292".into()),
            ],
            body: futures::stream::once(async move { Ok(body.into()) }).boxed(),
        })
    }
}

#[tokio::test]
async fn gateway_cache_policy_reaches_official_openrouter_chat_route() {
    let mock = Arc::new(Mock::default());
    let client = LlmClientBuilder::with_transport(
        mock.clone(),
        &[profile("openrouter", "https://openrouter.ai/api/v1")],
    )
    .with_region(Region::International)
    .build()
    .unwrap();
    client
        .chat()
        .complete(
            &request(),
            &RequestOptions {
                openrouter_response_cache: Some(OpenRouterResponseCache::Enabled {
                    ttl_seconds: Some(600),
                    refresh: true,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url,
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert!(requests[0]
        .headers
        .contains(&("X-OpenRouter-Cache".into(), "true".into())));
    assert!(requests[0]
        .headers
        .contains(&("X-OpenRouter-Cache-TTL".into(), "600".into())));
    assert!(requests[0]
        .headers
        .contains(&("X-OpenRouter-Cache-Clear".into(), "true".into())));
}

#[tokio::test]
async fn openrouter_response_cache_policy_reaches_responses_route_and_projects_status() {
    let mock = Arc::new(ResponsesRouteMock::default());
    let mut responses_profile = profile("openrouter", "https://openrouter.ai/api/v1");
    responses_profile.protocol = ProtocolFamily::OpenAiResponses;
    let client = LlmClientBuilder::with_transport(mock.clone(), &[responses_profile])
        .with_region(Region::International)
        .build()
        .unwrap();

    let response = client
        .chat()
        .complete(
            &request(),
            &RequestOptions {
                openrouter_response_cache: Some(OpenRouterResponseCache::Enabled {
                    ttl_seconds: Some(900),
                    refresh: false,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "https://openrouter.ai/api/v1/responses");
    assert!(requests[0]
        .headers
        .contains(&("X-OpenRouter-Cache".into(), "true".into())));
    assert!(requests[0]
        .headers
        .contains(&("X-OpenRouter-Cache-TTL".into(), "900".into())));
    drop(requests);

    let observation = response.response_cache.unwrap();
    assert_eq!(observation.status, ResponseCacheStatus::Hit);
    assert_eq!(observation.age_seconds, Some(8));
    assert_eq!(observation.ttl_seconds, Some(292));
}

#[tokio::test]
async fn gateway_cache_policy_rejects_other_provider_before_network() {
    let mock = Arc::new(Mock::default());
    let client = LlmClientBuilder::with_transport(
        mock.clone(),
        &[profile("openai", "https://api.openai.com/v1")],
    )
    .with_region(Region::International)
    .build()
    .unwrap();
    assert!(matches!(
        client
            .chat()
            .complete(
                &request(),
                &RequestOptions {
                    openrouter_response_cache: Some(OpenRouterResponseCache::Disabled),
                    ..Default::default()
                },
            )
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}
