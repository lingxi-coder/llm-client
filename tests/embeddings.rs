use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{embeddings::*, protocol::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
struct Mock {
    body: Value,
    requests: Mutex<Vec<HttpRequest>>,
    stalled: bool,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        if self.stalled {
            futures::future::pending::<()>().await;
        }
        let bytes = serde_json::to_vec(&self.body).unwrap().into();
        Ok(StreamResponse {
            status: 200,
            headers: vec![("x-request-id".into(), "req-test".into())],
            body: futures::stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}
fn setup(api: EmbeddingApi, body: Value, stalled: bool) -> (LlmClient, Arc<Mock>) {
    let profile:ProviderProfile=serde_json::from_value(json!({"provider_id":"test","profile_name":"test","protocol":"open_ai_chat","base_url":"https://chat.invalid","auth":"none","embeddings":{"mode":"enabled","value":{"api":api,"endpoint":if api==EmbeddingApi::Gemini{"https://embedding.invalid/models/{model}:batchEmbedContents"}else{"https://embedding.invalid/embed"},"models_endpoint":if api==EmbeddingApi::OpenRouter{Some("https://embedding.invalid/embeddings/models")}else{None},"auth":{"type":"bearer"},"max_inputs":2}}})).unwrap();
    let mock = Arc::new(Mock {
        body,
        requests: Mutex::new(vec![]),
        stalled,
    });
    (
        LlmClientBuilder::with_transport(mock.clone(), &[profile])
            .with_region(Region::International)
            .build()
            .unwrap(),
        mock,
    )
}
fn setup_unbounded(
    provider_id: &str,
    api: EmbeddingApi,
    endpoint: &str,
    body: Value,
) -> (LlmClient, Arc<Mock>) {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":provider_id, "profile_name":"unbounded",
        "base_url":"https://chat.invalid", "protocol":"open_ai_chat",
        "auth":"none",
        "embeddings":{"mode":"enabled","value":{
            "api":api, "endpoint":endpoint, "auth":{"type":"bearer"}
        }}
    }))
    .unwrap();
    let mock = Arc::new(Mock {
        body,
        requests: Mutex::new(vec![]),
        stalled: false,
    });
    (
        LlmClientBuilder::with_transport(mock.clone(), &[profile])
            .with_region(Region::International)
            .build()
            .unwrap(),
        mock,
    )
}
fn request() -> EmbeddingRequest {
    EmbeddingRequest {
        model: "embedding-test".into(),
        input: vec!["a".into(), "b".into()],
        dimensions: Some(2),
        task: None,
    }
}
fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("secret".into())),
        ..Default::default()
    }
}
fn qwen_response_body(input_count: usize, dimensions: usize) -> Value {
    let embeddings = (0..input_count)
        .map(|text_index| {
            json!({
                "text_index":text_index,
                "embedding":vec![0.0; dimensions]
            })
        })
        .collect::<Vec<_>>();
    json!({"output":{"embeddings":embeddings}})
}

fn openai_response_body(input_count: usize, dimensions: usize) -> Value {
    let data = (0..input_count)
        .map(|index| json!({"index":index,"embedding":vec![0.0; dimensions]}))
        .collect::<Vec<_>>();
    json!({"data":data})
}

#[tokio::test]
async fn service_only_profile_can_embed_without_chat_models() {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"test", "profile_name":"embedding-only",
        "base_url":"https://unused.invalid/v1", "protocol":"open_ai_chat",
        "auth":"none", "chat_enabled":false,
        "embeddings":{"mode":"enabled","value":{
            "api":"open_ai", "endpoint":"https://embedding.invalid/embed",
            "auth":{"type":"bearer"}, "max_inputs":2
        }}
    }))
    .unwrap();
    let mock = Arc::new(Mock {
        body: json!({"data":[{"index":0,"embedding":[1.0,2.0]}]}),
        requests: Mutex::new(vec![]),
        stalled: false,
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    let response = client
        .embeddings()
        .embed(
            "embedding-only",
            &EmbeddingRequest {
                model: "embedding-test".into(),
                input: vec!["hello".into()],
                dimensions: Some(2),
                task: None,
            },
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(response.vectors[0].values, vec![1.0, 2.0]);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
    assert!(client.resolve("embedding-test").is_err());
}

#[test]
fn service_only_profile_rejects_chat_models() {
    let mut profile = lingxi_llm_client::presets::builtin()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == "openai")
        .unwrap();
    profile.chat_enabled = false;
    let error = LlmClientBuilder::new(&[profile])
        .unwrap()
        .with_region(Region::International)
        .build()
        .err()
        .expect("Chat model rows must be rejected");
    assert!(matches!(error, BuildError::InvalidService { .. }));
}

#[test]
fn openrouter_preset_uses_its_own_embedding_adapter() {
    let presets = lingxi_llm_client::presets::builtin().unwrap();
    let profile = presets
        .iter()
        .find(|profile| profile.profile_name == "openrouter")
        .unwrap();
    assert!(matches!(
        &profile.embeddings,
        ServiceSetting::Enabled(route) if route.api == EmbeddingApi::OpenRouter
            && route.endpoint == "https://openrouter.ai/api/v1/embeddings"
            && route.models_endpoint.as_deref() == Some("https://openrouter.ai/api/v1/embeddings/models")
    ));
}

#[tokio::test]
async fn openrouter_model_directory_pages_and_preserves_native_metadata() {
    let (client, mock) = setup(
        EmbeddingApi::OpenRouter,
        json!({"data":[
            {"id":"vendor/text","name":"Text","context_length":8192,
             "architecture":{"input_modalities":["text"]},"pricing":{"prompt":"0.1"}},
            {"id":"vendor/multi","name":"Multi","context_length":4096,
             "architecture":{"input_modalities":["text","image"]}}
        ],"total_count":3}),
        false,
    );
    let page = client
        .embeddings()
        .list_models("test", 0, 2, &options())
        .await
        .unwrap();
    assert_eq!(page.models.len(), 2);
    assert_eq!(page.models[0].id, "vendor/text");
    assert_eq!(page.models[0].native["pricing"]["prompt"], "0.1");
    assert_eq!(page.models[1].input_modalities, ["text", "image"]);
    assert_eq!(page.total_count, Some(3));
    assert_eq!(page.next_offset, Some(2));
    assert_eq!(page.native["total_count"], 3);
    {
        let calls = mock.requests.lock().unwrap();
        assert_eq!(
            calls[0].url,
            "https://embedding.invalid/embeddings/models?offset=0&limit=2"
        );
        assert!(calls[0]
            .headers
            .iter()
            .any(|(key, value)| key == "authorization" && value == "Bearer secret"));
    }
    assert!(matches!(
        client
            .embeddings()
            .list_models("test", 0, 0, &options())
            .await,
        Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn model_directory_rejects_inconsistent_pagination() {
    let (client, mock) = setup(
        EmbeddingApi::OpenRouter,
        json!({"data":[{"id":"vendor/text","architecture":{"input_modalities":["text"]}}],
            "total_count":0}),
        false,
    );
    assert!(matches!(
        client
            .embeddings()
            .list_models("test", 0, 1, &options())
            .await,
        Err(EmbeddingError::InvalidResponse(_))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn independent_route_auth_dimensions_and_out_of_order_vectors() {
    let (client, mock) = setup(
        EmbeddingApi::OpenAi,
        json!({"model":"executed-model","usage":{"prompt_tokens":4,"total_tokens":4},"data":[{"index":1,"embedding":[3.0,4.0]},{"index":0,"embedding":[1.0,2.0]}]}),
        false,
    );
    let response = client
        .embeddings()
        .embed("test", &request(), &options())
        .await
        .unwrap();
    assert_eq!(response.vectors[0].values, vec![1.0, 2.0]);
    assert_eq!(response.model.as_deref(), Some("executed-model"));
    assert_eq!(response.usage.state, UsageState::Complete);
    assert_eq!(response.request_id.as_deref(), Some("req-test"));
    let calls = mock.requests.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].url, "https://embedding.invalid/embed");
    assert!(calls[0]
        .headers
        .iter()
        .any(|(k, v)| k == "authorization" && v == "Bearer secret"));
    let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
    assert_eq!(body["dimensions"], 2);
    assert_eq!(body["encoding_format"], "float");
}
#[tokio::test]
async fn gemini_and_qwen_preserve_task_type_and_wire_shape() {
    for api in [EmbeddingApi::Gemini, EmbeddingApi::Qwen] {
        let body = if api == EmbeddingApi::Gemini {
            json!({"embeddings":[{"values":[1,2]},{"values":[3,4]}]})
        } else {
            json!({"output":{"embeddings":[{"text_index":0,"embedding":[1,2]},{"text_index":1,"embedding":[3,4]}]}})
        };
        let (client, mock) = setup(api, body, false);
        let mut req = request();
        req.task = Some(EmbeddingTask::RetrievalQuery);
        let response = client
            .embeddings()
            .embed("test", &req, &options())
            .await
            .unwrap();
        assert_eq!(response.usage.state, UsageState::Missing);
        let calls = mock.requests.lock().unwrap();
        let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
        if api == EmbeddingApi::Gemini {
            assert!(calls[0]
                .url
                .contains("models/embedding-test:batchEmbedContents"));
            assert_eq!(
                body["requests"][0]["embedContentConfig"]["taskType"],
                "RETRIEVAL_QUERY"
            );
        } else {
            assert_eq!(body["parameters"]["text_type"], "query");
        }
    }
}

#[tokio::test]
async fn openrouter_uses_its_documented_input_type_and_float_vectors() {
    let (client, mock) = setup(
        EmbeddingApi::OpenRouter,
        json!({"model":"vendor/embed","usage":{"prompt_tokens":4,"total_tokens":4},
            "data":[{"index":1,"embedding":[3.0,4.0]},{"index":0,"embedding":[1.0,2.0]}]}),
        false,
    );
    let mut req = request();
    req.task = Some(EmbeddingTask::RetrievalDocument);
    let response = client
        .embeddings()
        .embed("test", &req, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors[0].values, vec![1.0, 2.0]);
    assert_eq!(response.usage.state, UsageState::Complete);
    {
        let calls = mock.requests.lock().unwrap();
        let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
        assert_eq!(body["input_type"], "search_document");
        assert_eq!(body["encoding_format"], "float");
        assert_eq!(body["dimensions"], 2);
    }
    req.task = Some(EmbeddingTask::Classification);
    assert!(matches!(
        client.embeddings().embed("test", &req, &options()).await,
        Err(EmbeddingError::Llm(LlmError::UnsupportedCapability { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn malformed_vectors_fail_without_normalization_or_retries() {
    for data in [
        json!([{ "index":0,"embedding":[1,2]}]),
        json!([{ "index":0,"embedding":[1,2]},{"index":0,"embedding":[3,4]}]),
        json!([{ "index":0,"embedding":[1,2]},{"index":1,"embedding":[3]}]),
        json!([{ "index":0,"embedding":[1,2]},{"index":2,"embedding":[3,4]}]),
        json!([{ "index":0,"embedding":[1,2]},{"index":1,"embedding":[null,4]}]),
    ] {
        let (client, mock) = setup(EmbeddingApi::OpenAi, json!({"data":data}), false);
        assert!(matches!(
            client
                .embeddings()
                .embed("test", &request(), &options())
                .await,
            Err(EmbeddingError::InvalidResponse(_))
        ));
        assert_eq!(mock.requests.lock().unwrap().len(), 1);
    }
}
#[tokio::test]
async fn preflight_has_no_network_side_effects_and_custom_transport_obeys_timeout() {
    let (client, mock) = setup(EmbeddingApi::OpenAi, json!({}), false);
    let mut req = request();
    req.input.push("c".into());
    assert!(client
        .embeddings()
        .embed("test", &req, &options())
        .await
        .is_err());
    assert!(mock.requests.lock().unwrap().is_empty());
    let (client, mock) = setup(EmbeddingApi::OpenAi, json!({}), true);
    let opts = RequestOptions {
        total_timeout: Some(std::time::Duration::from_millis(5)),
        ..options()
    };
    assert!(matches!(
        client.embeddings().embed("test", &request(), &opts).await,
        Err(EmbeddingError::Llm(LlmError::TransportTimeout { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn qwen_known_model_dimensions_and_batch_sizes_are_enforced_before_transport() {
    let cases = [
        ("qwen3.7-text-embedding", 21, None),
        ("qwen3.7-text-embedding-flash", 21, None),
        ("text-embedding-v4", 11, None),
        ("text-embedding-v3", 11, None),
        ("text-embedding-v2", 26, None),
        ("text-embedding-v1", 26, None),
        ("qwen3.7-text-embedding", 1, Some(128)),
        ("qwen3.7-text-embedding-flash", 1, Some(2048)),
        ("text-embedding-v4", 1, Some(2560)),
        ("text-embedding-v3", 1, Some(1536)),
        ("text-embedding-v1", 1, Some(1536)),
        ("text-embedding-v2", 1, Some(1536)),
    ];

    for (model, input_count, dimensions) in cases {
        let (client, mock) = setup_unbounded(
            "qwen",
            EmbeddingApi::Qwen,
            "https://embedding.invalid/embed",
            json!({}),
        );
        let request = EmbeddingRequest {
            model: model.into(),
            input: vec!["x".into(); input_count],
            dimensions,
            task: None,
        };
        assert!(
            matches!(
                client
                    .embeddings()
                    .embed("unbounded", &request, &options())
                    .await,
                Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
            ),
            "unexpectedly accepted Qwen model limits for {model}"
        );
        assert!(mock.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn openai_first_party_batch_limit_is_exact_and_gateway_is_not_inherited() {
    let openai_endpoint = "https://api.openai.com/v1/embeddings";
    let (client, mock) =
        setup_unbounded("openai", EmbeddingApi::OpenAi, openai_endpoint, json!({}));
    let too_many = EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: vec!["x".into(); 2049],
        dimensions: None,
        task: None,
    };
    assert!(matches!(
        client
            .embeddings()
            .embed("unbounded", &too_many, &options())
            .await,
        Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());

    let (client, mock) = setup_unbounded(
        "openai",
        EmbeddingApi::OpenAi,
        openai_endpoint,
        openai_response_body(2048, 1),
    );
    let boundary = EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: vec!["x".into(); 2048],
        dimensions: None,
        task: None,
    };
    let response = client
        .embeddings()
        .embed("unbounded", &boundary, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors.len(), 2048);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let gateway_endpoint = "https://openai-compatible.example/v1/embeddings";
    let (client, mock) = setup_unbounded(
        "openai",
        EmbeddingApi::OpenAi,
        gateway_endpoint,
        openai_response_body(2049, 1),
    );
    let response = client
        .embeddings()
        .embed("unbounded", &too_many, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors.len(), 2049);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let spoofed_host = "https://api.openai.com.evil.example/v1/embeddings";
    let (client, mock) = setup_unbounded(
        "openai",
        EmbeddingApi::OpenAi,
        spoofed_host,
        openai_response_body(2049, 1),
    );
    let response = client
        .embeddings()
        .embed("unbounded", &too_many, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors.len(), 2049);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let mismatched_provider = "https://api.openai.com/v1/embeddings";
    let (client, mock) = setup_unbounded(
        "openai-compatible",
        EmbeddingApi::OpenAi,
        mismatched_provider,
        openai_response_body(2049, 1),
    );
    let response = client
        .embeddings()
        .embed("unbounded", &too_many, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors.len(), 2049);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn openai_dimensions_parameter_is_rejected_for_ada_only_on_first_party_route() {
    let (client, mock) = setup_unbounded(
        "openai",
        EmbeddingApi::OpenAi,
        "https://api.openai.com/v1/embeddings",
        json!({}),
    );
    let ada_dimensions = EmbeddingRequest {
        model: "text-embedding-ada-002".into(),
        input: vec!["x".into()],
        dimensions: Some(2),
        task: None,
    };
    assert!(matches!(
        client
            .embeddings()
            .embed("unbounded", &ada_dimensions, &options())
            .await,
        Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());

    let (client, mock) = setup_unbounded(
        "openai",
        EmbeddingApi::OpenAi,
        "https://openai-compatible.example/v1/embeddings",
        openai_response_body(1, 2),
    );
    let response = client
        .embeddings()
        .embed("unbounded", &ada_dimensions, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors[0].values.len(), 2);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let (client, mock) = setup_unbounded(
        "openai",
        EmbeddingApi::OpenAi,
        "https://api.openai.com/v1/embeddings",
        openai_response_body(1, 2),
    );
    let future_model = EmbeddingRequest {
        model: "text-embedding-future".into(),
        input: vec!["x".into()],
        dimensions: Some(2),
        task: None,
    };
    let response = client
        .embeddings()
        .embed("unbounded", &future_model, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors[0].values.len(), 2);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn qwen_documented_batch_and_dimension_boundary_is_accepted() {
    let (client, mock) = setup_unbounded(
        "qwen",
        EmbeddingApi::Qwen,
        "https://embedding.invalid/embed",
        qwen_response_body(10, 64),
    );
    let request = EmbeddingRequest {
        model: "text-embedding-v4".into(),
        input: vec!["x".into(); 10],
        dimensions: Some(64),
        task: None,
    };
    let response = client
        .embeddings()
        .embed("unbounded", &request, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors.len(), 10);
    assert_eq!(response.vectors[0].values.len(), 64);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn glm_embedding_3_limits_apply_only_to_its_official_embedding_route() {
    let glm_endpoint = "https://open.bigmodel.cn/api/paas/v4/embeddings";
    let (client, mock) = setup_unbounded("zhipu", EmbeddingApi::OpenAi, glm_endpoint, json!({}));
    for (input_count, dimensions) in [(65, Some(256)), (1, Some(1536))] {
        let request = EmbeddingRequest {
            model: "embedding-3".into(),
            input: vec!["x".into(); input_count],
            dimensions,
            task: None,
        };
        assert!(matches!(
            client
                .embeddings()
                .embed("unbounded", &request, &options())
                .await,
            Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
        ));
    }
    assert!(mock.requests.lock().unwrap().is_empty());

    let glm_data = (0..64)
        .map(|index| json!({"index":index,"embedding":vec![0.0; 256]}))
        .collect::<Vec<_>>();
    let (client, mock) = setup_unbounded(
        "zhipu",
        EmbeddingApi::OpenAi,
        glm_endpoint,
        json!({"data":glm_data}),
    );
    let request = EmbeddingRequest {
        model: "embedding-3".into(),
        input: vec!["x".into(); 64],
        dimensions: Some(256),
        task: None,
    };
    let response = client
        .embeddings()
        .embed("unbounded", &request, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors.len(), 64);
    assert_eq!(response.vectors[0].values.len(), 256);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let generic_data = (0..65)
        .map(|index| json!({"index":index,"embedding":[1.0,2.0]}))
        .collect::<Vec<_>>();
    let (client, mock) = setup_unbounded(
        "test",
        EmbeddingApi::OpenAi,
        "https://embedding.invalid/embed",
        json!({"data":generic_data}),
    );
    let request = EmbeddingRequest {
        model: "embedding-3".into(),
        input: vec!["x".into(); 65],
        dimensions: Some(2),
        task: None,
    };
    let response = client
        .embeddings()
        .embed("unbounded", &request, &options())
        .await
        .unwrap();
    assert_eq!(response.vectors.len(), 65);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}
