use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{
    embeddings::EmbeddingError,
    presets,
    protocol::{LlmError, ProviderProfile, Region, Secret, ServiceSetting},
    providers::google::embeddings::{GeminiEmbeddingModelListQuery, GeminiEmbeddingPageToken},
    HttpRequest, LlmClient, LlmClientBuilder, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

struct Reply {
    status: u16,
    body: Value,
}

struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
    stalled: bool,
}

impl Mock {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            stalled: false,
        }
    }

    fn stalled() -> Self {
        Self {
            replies: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            stalled: true,
        }
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        if self.stalled {
            futures::future::pending::<()>().await;
        }
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected Gemini model-directory request");
        let bytes = serde_json::to_vec(&reply.body).unwrap().into();
        Ok(StreamResponse {
            status: reply.status,
            headers: vec![("x-request-id".into(), "catalog-request".into())],
            body: futures::stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}

fn reply(body: Value) -> Reply {
    Reply { status: 200, body }
}

fn model(resource_name: &str, id: &str, methods: Value) -> Value {
    json!({
        "name": resource_name,
        "baseModelId": id,
        "version": "001",
        "displayName": format!("Display {id}"),
        "description": "provider description",
        "inputTokenLimit": 2048,
        "outputTokenLimit": 3072,
        "supportedGenerationMethods": methods,
        "futureMetadata": {"preserved": true}
    })
}

fn gemini_profile() -> ProviderProfile {
    presets::builtin()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == "gemini")
        .expect("built-in Gemini profile")
}

fn client(mock: Arc<Mock>) -> LlmClient {
    LlmClientBuilder::with_transport(mock, &[gemini_profile()])
        .with_region(Region::International)
        .build()
        .unwrap()
}

fn options(scope: Option<&str>) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("api-key".into())),
        account_scope: scope.map(str::to_owned),
        ..Default::default()
    }
}

fn catalog_calls(mock: &Mock) -> Vec<HttpRequest> {
    mock.requests.lock().unwrap().clone()
}

fn has_header(request: &HttpRequest, name: &str, value: &str) -> bool {
    request
        .headers
        .iter()
        .any(|(key, actual)| key.eq_ignore_ascii_case(name) && actual == value)
}

#[tokio::test]
async fn gemini_model_directory_filters_by_explicit_method_and_preserves_native_fields() {
    let mock = Arc::new(Mock::new([reply(json!({
        "models": [
            model("models/gemini-3.8-flash", "gemini-3.8-flash", json!(["generateContent"])),
            model("models/gemini-embedding-001-002", "gemini-embedding-001", json!(["embedContent"])),
            model("models/gemini-embedding-2", "gemini-embedding-2", json!(["embedContent", "countTokens"]))
        ],
        "nextPageToken": "opaque+/cursor="
    }))]));
    let client = client(mock.clone());

    let page = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery {
                page_size: Some(1500),
                page_token: None,
            },
            &options(Some("project-a")),
        )
        .await
        .unwrap();

    assert_eq!(page.models.len(), 2);
    assert_eq!(
        page.models[0].resource_name,
        "models/gemini-embedding-001-002"
    );
    assert_eq!(page.models[0].id, "gemini-embedding-001-002");
    assert_eq!(page.models[0].base_model_id, "gemini-embedding-001");
    assert_eq!(page.models[0].input_token_limit, Some(2048));
    assert_eq!(page.models[0].output_token_limit, Some(3072));
    assert_eq!(page.models[0].native["futureMetadata"]["preserved"], true);
    assert_eq!(page.native["models"].as_array().unwrap().len(), 3);
    let cursor = page.next_page_token.unwrap();
    assert_eq!(cursor.page_size(), Some(1500));
    assert!(!format!("{cursor:?}").contains("opaque+/cursor="));

    let calls = catalog_calls(&mock);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].method, "GET");
    assert_eq!(
        calls[0].url,
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1500"
    );
    assert!(has_header(&calls[0], "x-goog-api-key", "api-key"));
}

#[tokio::test]
async fn empty_filtered_pages_keep_native_next_token_and_default_page_shape() {
    let mock = Arc::new(Mock::new([
        reply(json!({
            "models": [model("models/gemini-3.8-flash", "gemini-3.8-flash", json!(["generateContent"]))],
            "nextPageToken": "page/one+="
        })),
        reply(json!({
            "models": [model("models/gemini-embedding-2", "gemini-embedding-2", json!(["embedContent"]))]
        })),
    ]));
    let client = client(mock.clone());

    let first = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
        .await
        .unwrap();
    assert!(first.models.is_empty());
    let cursor = first.next_page_token.unwrap();
    assert_eq!(cursor.page_size(), None);
    let second = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery {
                page_size: None,
                page_token: Some(cursor),
            },
            &options(None),
        )
        .await
        .unwrap();
    assert_eq!(second.models[0].id, "gemini-embedding-2");
    assert!(second.next_page_token.is_none());

    let calls = catalog_calls(&mock);
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0].url,
        "https://generativelanguage.googleapis.com/v1beta/models"
    );
    assert_eq!(
        calls[1].url,
        "https://generativelanguage.googleapis.com/v1beta/models?pageToken=page%2Fone%2B%3D"
    );
}

#[tokio::test]
async fn page_token_binds_identity_and_exact_page_size_shape_before_auth() {
    let mock = Arc::new(Mock::new([reply(json!({
        "models": [], "nextPageToken": "page-a"
    }))]));
    let client = client(mock.clone());
    let first = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery::default(),
            &options(Some("project-a")),
        )
        .await
        .unwrap();
    let cursor = first.next_page_token.unwrap();

    let mut no_credential = options(Some("project-a"));
    no_credential.credential = None;
    let mismatch = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery {
                page_size: Some(50),
                page_token: Some(cursor.clone()),
            },
            &no_credential,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        mismatch,
        EmbeddingError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(catalog_calls(&mock).len(), 1);

    let scope_mismatch = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery {
                page_size: None,
                page_token: Some(cursor),
            },
            &options(Some("project-b")),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        scope_mismatch,
        EmbeddingError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(catalog_calls(&mock).len(), 1);

    let blank_scope = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery::default(),
            &RequestOptions {
                credential: None,
                account_scope: Some("   ".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(
        blank_scope,
        EmbeddingError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(catalog_calls(&mock).len(), 1);
}

#[tokio::test]
async fn a_repeated_next_page_token_is_rejected() {
    let mock = Arc::new(Mock::new([
        reply(json!({"models": [], "nextPageToken": "token-a"})),
        reply(json!({"models": [], "nextPageToken": "token-a"})),
    ]));
    let client = client(mock.clone());
    let first = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
        .await
        .unwrap();
    let repeated = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery {
                page_token: first.next_page_token,
                ..Default::default()
            },
            &options(None),
        )
        .await
        .unwrap_err();
    assert!(matches!(repeated, EmbeddingError::InvalidResponse(_)));
    assert_eq!(catalog_calls(&mock).len(), 2);
}

#[tokio::test]
async fn empty_or_missing_next_page_token_is_terminal_but_wrong_type_is_invalid() {
    for body in [
        json!({"models": []}),
        json!({"models": [], "nextPageToken": ""}),
    ] {
        let mock = Arc::new(Mock::new([reply(body)]));
        let page = client(mock)
            .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
            .unwrap()
            .embeddings()
            .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
            .await
            .unwrap();
        assert!(page.next_page_token.is_none());
    }

    for malformed in [json!(false), json!(7), json!({"token":"x"})] {
        let mock = Arc::new(Mock::new([reply(json!({
            "models": [], "nextPageToken": malformed
        }))]));
        let result = client(mock)
            .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
            .unwrap()
            .embeddings()
            .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
            .await;
        assert!(matches!(result, Err(EmbeddingError::InvalidResponse(_))));
    }
}

#[tokio::test]
async fn malformed_native_model_fields_are_not_silently_dropped() {
    let mut malformed = model(
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        json!(["embedContent"]),
    );
    malformed["inputTokenLimit"] = json!("2048");
    let mock = Arc::new(Mock::new([reply(json!({"models": [malformed]}))]));
    let result = client(mock)
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
        .await;
    assert!(matches!(result, Err(EmbeddingError::InvalidResponse(_))));

    let malformed_methods = model(
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        json!(["embedContent", true]),
    );
    let mock = Arc::new(Mock::new([reply(json!({"models": [malformed_methods]}))]));
    let result = client(mock)
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
        .await;
    assert!(matches!(result, Err(EmbeddingError::InvalidResponse(_))));

    let mut malformed_display_name = model(
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        json!(["embedContent"]),
    );
    malformed_display_name["displayName"] = json!(false);
    let mock = Arc::new(Mock::new([reply(json!({
        "models": [malformed_display_name]
    }))]));
    let result = client(mock)
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
        .await;
    assert!(matches!(result, Err(EmbeddingError::InvalidResponse(_))));
}

#[tokio::test]
async fn gemini_models_get_preserves_resource_name_and_returns_bare_embed_id() {
    let mock = Arc::new(Mock::new([reply(model(
        "models/gemini-embedding-2",
        "gemini-embedding-2",
        json!(["embedContent"]),
    ))]));
    let result = client(mock.clone())
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .get_model("models/gemini-embedding-2", &options(Some("project-a")))
        .await
        .unwrap();
    assert_eq!(result.resource_name, "models/gemini-embedding-2");
    assert_eq!(result.id, "gemini-embedding-2");
    assert_eq!(
        catalog_calls(&mock)[0].url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2"
    );

    let mock = Arc::new(Mock::new([reply(model(
        "models/gemini-3.8-flash",
        "gemini-3.8-flash",
        json!(["generateContent"]),
    ))]));
    let unsupported = client(mock)
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .get_model("models/gemini-3.8-flash", &options(None))
        .await
        .unwrap_err();
    assert!(matches!(
        unsupported,
        EmbeddingError::Llm(LlmError::UnsupportedCapability { .. })
    ));

    let mock = Arc::new(Mock::new([]));
    let invalid_resource = client(mock.clone())
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .get_model("models/invalid+id", &RequestOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(
        invalid_resource,
        EmbeddingError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert!(catalog_calls(&mock).is_empty());
}

#[tokio::test]
async fn gemini_model_directory_rejects_response_name_mismatch_and_oversized_pages() {
    let mock = Arc::new(Mock::new([reply(model(
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        json!(["embedContent"]),
    ))]));
    let mismatch = client(mock)
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .get_model("models/gemini-embedding-2", &options(None))
        .await
        .unwrap_err();
    assert!(matches!(mismatch, EmbeddingError::InvalidResponse(_)));

    let too_many = (0..1001)
        .map(|index| {
            model(
                &format!("models/gemini-embedding-{index}"),
                &format!("gemini-embedding-{index}"),
                json!(["embedContent"]),
            )
        })
        .collect::<Vec<_>>();
    let mock = Arc::new(Mock::new([reply(json!({"models": too_many}))]));
    let result = client(mock)
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(
            &GeminiEmbeddingModelListQuery {
                page_size: Some(2000),
                page_token: None,
            },
            &options(None),
        )
        .await;
    assert!(matches!(result, Err(EmbeddingError::InvalidResponse(_))));
}

#[tokio::test]
async fn disabled_catalog_and_total_timeout_keep_preflight_contract() {
    let mut profile = gemini_profile();
    profile.embeddings = ServiceSetting::Disabled;
    let mock = Arc::new(Mock::new([]));
    let disabled_client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    let error = disabled_client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), &options(None))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        EmbeddingError::Llm(LlmError::UnsupportedCapability { .. })
    ));
    assert!(catalog_calls(&mock).is_empty());

    let stalled = Arc::new(Mock::stalled());
    let client = client(stalled.clone());
    let mut timed_options = options(None);
    timed_options.total_timeout = Some(Duration::from_millis(10));
    let error = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap()
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), &timed_options)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        EmbeddingError::Llm(LlmError::TransportTimeout { .. })
    ));
    assert_eq!(catalog_calls(&stalled).len(), 1);
}

#[test]
fn page_token_can_be_serialized_without_exposing_it_in_debug() {
    let token: GeminiEmbeddingPageToken = serde_json::from_value(json!({
        "token": "sensitive-page-token",
        "page_size": null,
        "provider_id": "google",
        "profile_name": "gemini",
        "embedding_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
        "models_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "region": "international",
        "account_scope": null
    }))
    .unwrap();
    assert!(!format!("{token:?}").contains("sensitive-page-token"));
    assert_eq!(
        serde_json::to_value(token).unwrap()["token"],
        "sensitive-page-token"
    );
}
