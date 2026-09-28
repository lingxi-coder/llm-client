use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::providers::openrouter::response_cache::OpenRouterResponseCache;
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
            "id":"msg-cache-test","type":"message","role":"assistant",
            "model":"anthropic/claude-sonnet-4",
            "content":[{"type":"text","text":"ok"}],
            "stop_reason":"end_turn","usage":{"input_tokens":0,"output_tokens":0}
        }))
        .unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![
                ("X-OpenRouter-Cache-Status".into(), "HIT".into()),
                ("X-OpenRouter-Cache-Age".into(), "9".into()),
                ("X-OpenRouter-Cache-TTL".into(), "291".into()),
                ("X-OpenRouter-Cache-Source-Id".into(), "gen-source".into()),
            ],
            body: futures::stream::once(async move { Ok(body.into()) }).boxed(),
        })
    }
}

#[tokio::test]
async fn openrouter_messages_uses_documented_route_and_cache_contract() {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"openrouter", "profile_name":"openrouter-messages",
        "base_url":"https://openrouter.ai/api/v1",
        "protocol":"anthropic_messages", "auth":"bearer",
        "models":[{"request_model":"anthropic/claude-sonnet-4","display_model":"Claude Sonnet 4","billing_model":"anthropic/claude-sonnet-4"}]
    }))
    .unwrap();
    let mock = Arc::new(Mock::default());
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    let request: ChatRequest = serde_json::from_value(json!({
        "model":"anthropic/claude-sonnet-4",
        "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
    }))
    .unwrap();

    let response = client
        .chat()
        .complete(
            &request,
            &RequestOptions {
                credential: Some(Secret::new("key".into())),
                openrouter_response_cache: Some(OpenRouterResponseCache::Enabled {
                    ttl_seconds: Some(300),
                    refresh: false,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sent = mock.requests.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url, "https://openrouter.ai/api/v1/messages");
    assert!(sent[0]
        .headers
        .iter()
        .any(|(key, value)| key.eq_ignore_ascii_case("authorization") && value == "Bearer key"));
    assert!(sent[0]
        .headers
        .contains(&("X-OpenRouter-Cache".into(), "true".into())));
    assert!(sent[0]
        .headers
        .contains(&("X-OpenRouter-Cache-TTL".into(), "300".into())));
    let observation = response.response_cache.unwrap();
    assert_eq!(observation.status, ResponseCacheStatus::Hit);
    assert_eq!(observation.age_seconds, Some(9));
    assert_eq!(observation.ttl_seconds, Some(291));
    assert_eq!(
        observation.source_generation_id.as_deref(),
        Some("gen-source")
    );
}
