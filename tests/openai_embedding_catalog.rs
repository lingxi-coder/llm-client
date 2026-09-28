use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{
    embeddings::EmbeddingError,
    presets,
    protocol::{LlmError, ProviderProfile, Region, Secret, ServiceSetting},
    providers::openai::embeddings::OpenAiEmbeddingModelPage,
    HttpRequest, LlmClient, LlmClientBuilder, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

struct Mock {
    status: u16,
    body: Value,
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let bytes = serde_json::to_vec(&self.body).unwrap().into();
        Ok(StreamResponse {
            status: self.status,
            headers: vec![("x-request-id".into(), "openai-catalog-request".into())],
            body: futures::stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}

fn openai_profile() -> ProviderProfile {
    presets::builtin()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == "openai")
        .expect("built-in OpenAI profile")
}

fn client(profile: ProviderProfile, status: u16, body: Value) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        status,
        body,
        requests: Mutex::new(Vec::new()),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("api-key".into())),
        ..Default::default()
    }
}

fn model(id: &str) -> Value {
    json!({
        "id": id,
        "object": "model",
        "created": 1705948997,
        "owned_by": "openai",
        "futureMetadata": {"preserved": true}
    })
}

#[tokio::test]
async fn openai_model_directory_filters_documented_ids_and_keeps_all_native_rows() {
    let body = json!({
        "object": "list",
        "data": [
            model("text-embedding-3-small"),
            model("gpt-6-sol"),
            model("future-embedding-model"),
            model("text-embedding-ada-002")
        ],
        "futureListField": "preserved"
    });
    let (client, mock) = client(openai_profile(), 200, body);

    let page: OpenAiEmbeddingModelPage = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .embeddings()
        .list_models(&options())
        .await
        .unwrap();

    assert_eq!(page.models.len(), 2);
    assert_eq!(page.models[0].id, "text-embedding-3-small");
    assert_eq!(page.models[0].created, 1705948997);
    assert_eq!(page.models[0].owned_by, "openai");
    assert_eq!(page.models[0].native["futureMetadata"]["preserved"], true);
    assert_eq!(page.models[1].id, "text-embedding-ada-002");
    assert_eq!(page.native["data"].as_array().unwrap().len(), 4);
    assert_eq!(page.native["futureListField"], "preserved");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[0].url, "https://api.openai.com/v1/models");
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer api-key"
    }));
}

#[tokio::test]
async fn openai_catalog_requires_first_party_routes_before_transport() {
    let mut profile = openai_profile();
    let ServiceSetting::Enabled(route) = &mut profile.embeddings else {
        panic!("built-in OpenAI embeddings should be enabled");
    };
    route.endpoint = "https://openai-compatible.example/v1/embeddings".into();
    route.models_endpoint = Some("https://openai-compatible.example/v1/models".into());
    let (client, mock) = client(profile, 200, json!({}));

    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .embeddings()
            .list_models(&options())
            .await,
        Err(EmbeddingError::Llm(LlmError::UnsupportedCapability { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn openai_catalog_rejects_duplicate_model_ids() {
    let body = json!({
        "object": "list",
        "data": [model("text-embedding-3-small"), model("text-embedding-3-small")]
    });
    let (client, mock) = client(openai_profile(), 200, body);

    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .embeddings()
            .list_models(&options())
            .await,
        Err(EmbeddingError::InvalidResponse(_))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn openai_catalog_rejects_disabled_and_wrong_region_before_transport() {
    let mut disabled_profile = openai_profile();
    disabled_profile.embeddings = ServiceSetting::Disabled;
    let (disabled_client, disabled_mock) = client(disabled_profile, 200, json!({}));
    assert!(matches!(
        disabled_client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .embeddings()
            .list_models(&options())
            .await,
        Err(EmbeddingError::Llm(LlmError::UnsupportedCapability { .. }))
    ));
    assert!(disabled_mock.requests.lock().unwrap().is_empty());

    let mut wrong_region_profile = openai_profile();
    wrong_region_profile.regions = vec![Region::ChinaMainland];
    let (wrong_region_client, wrong_region_mock) = client(wrong_region_profile, 200, json!({}));
    assert!(matches!(
        wrong_region_client.provider::<lingxi_llm_client::providers::OpenAiClient>("openai"),
        Err(lingxi_llm_client::providers::ProviderBindingError::UnavailableRegion { .. })
    ));
    assert!(wrong_region_mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn openai_catalog_requires_credentials_before_transport() {
    let (client, mock) = client(openai_profile(), 200, json!({"object":"list","data":[]}));

    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .embeddings()
            .list_models(&RequestOptions::default())
            .await,
        Err(EmbeddingError::Llm(LlmError::Authentication { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}
