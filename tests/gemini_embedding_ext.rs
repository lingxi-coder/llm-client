use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{embeddings::*, protocol::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

struct Mock {
    body: Value,
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let bytes = serde_json::to_vec(&self.body).unwrap().into();
        Ok(StreamResponse {
            status: 200,
            headers: vec![("x-request-id".into(), "gemini-embedding-test".into())],
            body: futures::stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}

fn setup(body: Value) -> (LlmClient, Arc<Mock>) {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"gemini",
        "profile_name":"gemini-test",
        "protocol":"gemini_generate_content",
        "base_url":"https://chat.invalid",
        "auth":"none",
        "embeddings":{
            "mode":"enabled",
            "value":{
                "api":"gemini",
                "endpoint":"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
                "auth":{"type":"api_key","header":"x-goog-api-key"},
                "max_inputs":100
            }
        }
    }))
    .unwrap();
    let mock = Arc::new(Mock {
        body,
        requests: Mutex::new(vec![]),
    });
    (
        LlmClientBuilder::with_transport(mock.clone(), &[profile])
            .with_region(Region::International)
            .build()
            .unwrap(),
        mock,
    )
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("gemini-test-key".into())),
        ..Default::default()
    }
}

fn inline_media(mime_type: &str, data: &[u8]) -> GeminiEmbeddingMedia {
    GeminiEmbeddingMedia {
        mime_type: mime_type.into(),
        source: GeminiEmbeddingSource::Inline(data.to_vec()),
        duration_seconds: None,
        page_count: None,
    }
}

#[tokio::test]
async fn multimodal_parts_use_the_documented_batch_request_and_aggregate_one_vector() {
    let mut embedding = vec![0.0; 768];
    embedding[0] = 0.25;
    embedding[1] = 0.5;
    let (client, mock) = setup(json!({
        "embeddings":[{"values":embedding}],
        "usageMetadata":{"promptTokenCount":19,"totalTokenCount":19}
    }));
    let audio = GeminiEmbeddingMedia {
        mime_type: "audio/mpeg".into(),
        source: GeminiEmbeddingSource::FileUri(
            "https://generativelanguage.googleapis.com/v1beta/files/audio-1".into(),
        ),
        duration_seconds: Some(180),
        page_count: None,
    };
    let video = GeminiEmbeddingMedia {
        mime_type: "video/mp4".into(),
        source: GeminiEmbeddingSource::Inline(vec![0, 1, 2, 3]),
        duration_seconds: Some(120),
        page_count: None,
    };
    let request = GeminiMultimodalEmbeddingRequest {
        model: "gemini-embedding-2".into(),
        parts: vec![
            GeminiEmbeddingPart::Text("Describe this clip".into()),
            GeminiEmbeddingPart::Media(inline_media("image/png", &[137, 80, 78, 71])),
            GeminiEmbeddingPart::Media(audio),
            GeminiEmbeddingPart::Media(video),
        ],
        dimensions: Some(768),
    };
    let response = client
        .embeddings()
        .embed_gemini_multimodal("gemini-test", &request, &options())
        .await
        .unwrap();

    assert_eq!(response.vectors.len(), 1);
    assert_eq!(response.vectors[0].values.len(), 768);
    assert_eq!(&response.vectors[0].values[..2], [0.25, 0.5]);
    assert_eq!(response.usage.state, UsageState::Complete);
    assert_eq!(
        response.request_id.as_deref(),
        Some("gemini-embedding-test")
    );
    let calls = mock.requests.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents");
    assert!(calls[0]
        .headers
        .iter()
        .any(|(key, value)| { key == "x-goog-api-key" && value == "gemini-test-key" }));
    let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
    assert_eq!(body["requests"].as_array().unwrap().len(), 1);
    assert_eq!(body["requests"][0]["model"], "models/gemini-embedding-2");
    assert_eq!(
        body["requests"][0]["embedContentConfig"]["outputDimensionality"],
        768
    );
    assert_eq!(
        body["requests"][0]["embedContentConfig"]["autoTruncate"],
        false
    );
    assert!(body["requests"][0]["embedContentConfig"]
        .get("taskType")
        .is_none());
    let parts = body["requests"][0]["content"]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0]["text"], "Describe this clip");
    assert_eq!(parts[1]["inline_data"]["mime_type"], "image/png");
    assert_eq!(parts[1]["inline_data"]["data"], "iVBORw==");
    assert_eq!(parts[2]["file_data"]["mime_type"], "audio/mpeg");
    assert_eq!(
        parts[2]["file_data"]["file_uri"],
        "https://generativelanguage.googleapis.com/v1beta/files/audio-1"
    );
    assert_eq!(parts[3]["inline_data"]["mime_type"], "video/mp4");
}

#[tokio::test]
async fn model_specific_tasks_and_dimensions_fail_before_transport() {
    let (client, mock) = setup(json!({"embeddings":[{"values":vec![0.0; 128]}]}));
    let ordinary = EmbeddingRequest {
        model: "gemini-embedding-2".into(),
        input: vec!["text".into()],
        dimensions: None,
        task: Some(EmbeddingTask::RetrievalQuery),
    };
    assert!(matches!(
        client
            .embeddings()
            .embed("gemini-test", &ordinary, &options())
            .await,
        Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
    ));
    for dimensions in [1, 3073] {
        let request = GeminiMultimodalEmbeddingRequest {
            model: "gemini-embedding-2".into(),
            parts: vec![GeminiEmbeddingPart::Text("text".into())],
            dimensions: Some(dimensions),
        };
        assert!(matches!(
            client
                .embeddings()
                .embed_gemini_multimodal("gemini-test", &request, &options())
                .await,
            Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
        ));
    }
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn multimodal_media_limits_and_metadata_are_preflighted() {
    let (client, mock) = setup(json!({"embeddings":[{"values":vec![0.0; 128]}]}));
    let invalid_requests = [
        GeminiMultimodalEmbeddingRequest {
            model: "gemini-embedding-2".into(),
            parts: (0..7)
                .map(|_| GeminiEmbeddingPart::Media(inline_media("image/jpeg", &[1])))
                .collect(),
            dimensions: None,
        },
        GeminiMultimodalEmbeddingRequest {
            model: "gemini-embedding-2".into(),
            parts: vec![GeminiEmbeddingPart::Media(GeminiEmbeddingMedia {
                mime_type: "audio/wav".into(),
                source: GeminiEmbeddingSource::Inline(vec![1]),
                duration_seconds: Some(181),
                page_count: None,
            })],
            dimensions: None,
        },
        GeminiMultimodalEmbeddingRequest {
            model: "gemini-embedding-2".into(),
            parts: vec![GeminiEmbeddingPart::Media(GeminiEmbeddingMedia {
                mime_type: "video/quicktime".into(),
                source: GeminiEmbeddingSource::Inline(vec![1]),
                duration_seconds: Some(121),
                page_count: None,
            })],
            dimensions: None,
        },
        GeminiMultimodalEmbeddingRequest {
            model: "gemini-embedding-2".into(),
            parts: vec![GeminiEmbeddingPart::Media(GeminiEmbeddingMedia {
                mime_type: "application/pdf".into(),
                source: GeminiEmbeddingSource::Inline(vec![1]),
                duration_seconds: None,
                page_count: Some(7),
            })],
            dimensions: None,
        },
        GeminiMultimodalEmbeddingRequest {
            model: "gemini-embedding-2".into(),
            parts: vec![GeminiEmbeddingPart::Media(inline_media("image/webp", &[1]))],
            dimensions: None,
        },
        GeminiMultimodalEmbeddingRequest {
            model: "gemini-embedding-001".into(),
            parts: vec![GeminiEmbeddingPart::Text("text".into())],
            dimensions: None,
        },
    ];
    for request in invalid_requests {
        assert!(matches!(
            client
                .embeddings()
                .embed_gemini_multimodal("gemini-test", &request, &options())
                .await,
            Err(EmbeddingError::Llm(LlmError::InvalidRequest { .. }))
        ));
    }
    assert!(mock.requests.lock().unwrap().is_empty());
}
