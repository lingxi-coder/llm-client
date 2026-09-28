use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::providers::openrouter::response_cache::OpenRouterResponseCache;
use lingxi_llm_client::{embeddings::*, protocol::*, *};
use serde_json::json;
use std::sync::{Arc, Mutex};

struct Mock {
    requests: Mutex<Vec<HttpRequest>>,
    headers: Vec<(String, String)>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let bytes = serde_json::to_vec(&json!({
            "model":"vendor/embedding",
            "data":[{"index":0,"embedding":[1.0,2.0]}],
            "usage":{"prompt_tokens":2,"total_tokens":2}
        }))
        .unwrap()
        .into();
        Ok(StreamResponse {
            status: 200,
            headers: self.headers.clone(),
            body: futures::stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}

fn setup(provider: &str, endpoint: &str, headers: Vec<(String, String)>) -> (LlmClient, Arc<Mock>) {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":provider,
        "profile_name":"embedding-only",
        "base_url":"https://openrouter.ai/api/v1",
        "protocol":"open_ai_chat",
        "auth":"none",
        "chat_enabled":false,
        "embeddings":{"mode":"enabled","value":{
            "api":"open_router","endpoint":endpoint,
            "auth":{"type":"bearer"},"max_inputs":2
        }}
    }))
    .unwrap();
    let mock = Arc::new(Mock {
        requests: Mutex::new(vec![]),
        headers,
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}

fn request() -> EmbeddingRequest {
    EmbeddingRequest {
        model: "vendor/embedding".into(),
        input: vec!["hello".into()],
        dimensions: Some(2),
        task: None,
    }
}

fn options(ttl_seconds: u32) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("secret".into())),
        openrouter_response_cache: Some(OpenRouterResponseCache::Enabled {
            ttl_seconds: Some(ttl_seconds),
            refresh: true,
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn official_embeddings_route_sends_cache_headers_and_projects_hit() {
    let (client, mock) = setup(
        "openrouter",
        "https://openrouter.ai/api/v1/embeddings",
        vec![
            ("X-OpenRouter-Cache-Status".into(), "HIT".into()),
            ("X-OpenRouter-Cache-Age".into(), "15".into()),
            ("X-OpenRouter-Cache-TTL".into(), "585".into()),
            ("X-OpenRouter-Cache-Source-Id".into(), "generation-1".into()),
        ],
    );
    let response = client
        .embeddings()
        .embed("embedding-only", &request(), &options(600))
        .await
        .unwrap();
    let observation = response.response_cache.unwrap();
    assert_eq!(observation.status, ResponseCacheStatus::Hit);
    assert_eq!(observation.age_seconds, Some(15));
    assert_eq!(observation.ttl_seconds, Some(585));
    assert_eq!(
        observation.source_generation_id.as_deref(),
        Some("generation-1")
    );
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    for (name, value) in [
        ("X-OpenRouter-Cache", "true"),
        ("X-OpenRouter-Cache-TTL", "600"),
        ("X-OpenRouter-Cache-Clear", "true"),
    ] {
        assert!(requests[0]
            .headers
            .iter()
            .any(|(key, actual)| key == name && actual == value));
    }
}

#[tokio::test]
async fn unknown_status_is_not_an_observation() {
    let (client, _) = setup(
        "openrouter",
        "https://openrouter.ai/api/v1/embeddings",
        vec![("X-OpenRouter-Cache-Status".into(), "STALE".into())],
    );
    let response = client
        .embeddings()
        .embed("embedding-only", &request(), &options(600))
        .await
        .unwrap();
    assert!(response.response_cache.is_none());
}

#[tokio::test]
async fn invalid_ttl_and_non_official_route_fail_before_http() {
    for (provider, endpoint, ttl) in [
        ("openrouter", "https://openrouter.ai/api/v1/embeddings", 0),
        ("openrouter", "https://proxy.invalid/api/v1/embeddings", 600),
        ("other", "https://openrouter.ai/api/v1/embeddings", 600),
    ] {
        let (client, mock) = setup(provider, endpoint, vec![]);
        let error = client
            .embeddings()
            .embed("embedding-only", &request(), &options(ttl))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            EmbeddingError::Llm(
                LlmError::InvalidRequest { .. } | LlmError::UnsupportedCapability { .. }
            )
        ));
        assert!(mock.requests.lock().unwrap().is_empty());
    }
}
