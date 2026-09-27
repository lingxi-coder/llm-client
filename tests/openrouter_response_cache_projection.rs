use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{ChatRequest, LlmError, ProviderProfile, Region, ResponseCacheStatus},
    *,
};
use serde_json::json;
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[derive(Default)]
struct Mock {
    replies: Mutex<VecDeque<Reply>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        Ok(StreamResponse {
            status: 200,
            headers: reply.headers,
            body: futures::stream::once(async move { Ok(reply.body.into()) }).boxed(),
        })
    }
}

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openrouter",
        "profile_name":"openrouter",
        "base_url":"https://openrouter.ai/api/v1",
        "protocol":"open_ai_chat",
        "auth":"none",
        "models":[{"display_model":"test","request_model":"test","billing_model":"test"}]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"test",
        "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]
    }))
    .unwrap()
}

fn response_body(prompt_tokens: u64, completion_tokens: u64) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "id":"chatcmpl-cache-test",
        "model":"test",
        "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
        "usage":{
            "prompt_tokens":prompt_tokens,
            "completion_tokens":completion_tokens,
            "total_tokens":prompt_tokens + completion_tokens
        }
    }))
    .unwrap()
}

#[tokio::test]
async fn complete_and_stream_project_only_explicit_openrouter_cache_status_headers() {
    let mock = std::sync::Arc::new(Mock {
        replies: Mutex::new(
            vec![
                Reply {
                    headers: vec![
                        ("x-openrouter-cache-status".into(), "HIT".into()),
                        ("X-OpenRouter-Cache-Age".into(), "12".into()),
                        ("X-OpenRouter-Cache-TTL".into(), "288".into()),
                        ("X-OpenRouter-Cache-Source-Id".into(), "gen-original".into()),
                    ],
                    body: response_body(0, 0),
                },
                Reply {
                    headers: vec![
                        ("X-OpenRouter-Cache-Status".into(), "MISS".into()),
                        ("X-OpenRouter-Cache-TTL".into(), "300".into()),
                    ],
                    body: response_body(10, 2),
                },
                Reply {
                    headers: vec![("X-OpenRouter-Cache-Status".into(), "future-value".into())],
                    body: response_body(0, 0),
                },
                Reply {
                    headers: vec![("X-OpenRouter-Cache-Status".into(), "HIT".into())],
                    body: Vec::new(),
                },
            ]
            .into(),
        ),
    });
    let client = LlmClientBuilder::with_transport(mock, &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let options = RequestOptions {
        openrouter_response_cache: Some(OpenRouterResponseCache::Enabled {
            ttl_seconds: Some(300),
            refresh: false,
        }),
        ..Default::default()
    };

    let hit = client.chat().complete(&request(), &options).await.unwrap();
    let observation = hit.response_cache.unwrap();
    assert_eq!(observation.status, ResponseCacheStatus::Hit);
    assert_eq!(observation.age_seconds, Some(12));
    assert_eq!(observation.ttl_seconds, Some(288));
    assert_eq!(
        observation.source_generation_id.as_deref(),
        Some("gen-original")
    );
    assert_eq!(hit.usage.usage.unwrap().total(), 0);

    let miss = client.chat().complete(&request(), &options).await.unwrap();
    let observation = miss.response_cache.unwrap();
    assert_eq!(observation.status, ResponseCacheStatus::Miss);
    assert_eq!(observation.age_seconds, None);
    assert_eq!(observation.ttl_seconds, Some(300));
    assert_eq!(observation.source_generation_id, None);

    let unknown = client.chat().complete(&request(), &options).await.unwrap();
    assert!(unknown.response_cache.is_none());
    assert_eq!(unknown.usage.usage.unwrap().total(), 0);

    let stream = client.chat().stream(&request(), &options).await.unwrap();
    let observation = stream.response_cache().unwrap();
    assert_eq!(observation.status, ResponseCacheStatus::Hit);
    assert_eq!(observation.age_seconds, None);
}
