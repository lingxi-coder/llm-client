# Gemini Embeddings

The regular `EmbeddingRequest` supports Gemini's text embedding batch route. For one mixed text-and-media content item, use `GeminiMultimodalEmbeddingRequest` with `embed_gemini_multimodal`; Gemini Embedding 2 combines all supplied parts into one vector.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::embeddings::{
    GeminiEmbeddingMedia, GeminiEmbeddingPart, GeminiEmbeddingSource,
    GeminiMultimodalEmbeddingRequest,
};

async fn embed_image(
    client: &LlmClient,
    options: &RequestOptions,
    image_bytes: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
let input = GeminiMultimodalEmbeddingRequest {
    model: "gemini-embedding-2".into(),
    parts: vec![
        GeminiEmbeddingPart::Text("A red bicycle beside a tree".into()),
        GeminiEmbeddingPart::Media(GeminiEmbeddingMedia {
            mime_type: "image/jpeg".into(),
            source: GeminiEmbeddingSource::Inline(image_bytes),
            duration_seconds: None,
            page_count: None,
        }),
    ],
    dimensions: Some(768),
};

let result = client
    .embeddings()
    .embed_gemini_multimodal("gemini", &input, &options)
    .await?;
assert_eq!(result.vectors.len(), 1);
Ok(())
}
```

`GeminiEmbeddingSource::Inline` sends bytes as base64 in the documented `inline_data` field. `FileUri` passes a URI that the caller has already uploaded to the Gemini Files API; this client does not upload, download, poll, or refresh media. The request uses `batchEmbedContents` with one `EmbedContentRequest`, which preserves a single aggregated embedding for the supplied `Content.parts`.

Google documents `gemini-embedding-2` inputs as text, PNG/JPEG images, MP3/WAV audio, MP4/MOV video, and PDF. The documented limits are 8,192 input tokens, six images, 180 seconds of audio, 120 seconds of video, and one PDF of at most six pages. The output dimension range is 128–3,072; Google recommends 768, 1,536, or 3,072. The client validates media types, image/PDF counts, declared audio/video durations, PDF page counts, and output dimensions before sending. Durations and page counts are caller-supplied validation metadata; the client does not parse media files. Google remains authoritative for media validity and the 8,192-token limit. `autoTruncate` is set to `false` so oversized input is rejected by the provider instead of silently shortened. Videos are sampled by Google at up to 32 frames, and audio tracks in video files are not processed.

`gemini-embedding-2` does not accept `taskType`. For text-only tasks, include Google's recommended task instruction in a text part. For multimodal content, Google generally advises against adding a task prefix. `gemini-embedding-001` remains text-only, has a 2,048-token input limit, and supports `taskType`; it cannot be used with the multimodal method. Gemini Embedding 1 and 2 vectors belong to incompatible spaces, so do not compare them directly. Returned vector values are preserved without normalization.

## Discovering embedding models

The Gemini profile configures its own `models_endpoint`; discovery does not derive a URL from Chat configuration. `list_gemini_models` calls Google's `models.list` route and returns only rows whose provider-reported `supportedGenerationMethods` contains the exact `embedContent` method. Each `GeminiEmbeddingModel` retains the official `resource_name`, its bare resource-name suffix as `id` (ready for `EmbeddingRequest.model`), the separate provider-reported `base_model_id`, version, reported token limits and method list, and the full native object. Fields not published by this directory, including embedding dimensions and input modalities, are not inferred from model names.

`GeminiEmbeddingModelListQuery.page_size` is optional: `None` leaves `pageSize` off the URL and uses Google's default of 50. Positive values are sent as given; Google may return at most 1000 rows even for a larger request. A page's `next_page_token` is an opaque, serializable cursor rather than an offset. It preserves the original `pageSize` parameter shape, route, provider, profile, region and optional `RequestOptions.account_scope`; reusing it against another scope or explicitly changing the page size is rejected before authentication or HTTP. An omitted account scope remains unbound to a declared account identity, and the client does not claim that the label verifies which API key is in use. Empty or missing `nextPageToken` ends pagination. Filtering does not fetch extra pages: a page with no embedding models can still have a continuation.

`get_gemini_model` accepts the official resource name such as `models/gemini-embedding-2`, verifies the returned resource name, and rejects a resource unless its methods explicitly include `embedContent`. Listing and lookup preserve the enabled/disabled profile, region, per-call credential and total-timeout rules of other embedding operations.

```rust,no_run
use lingxi_llm_client::{embeddings::{EmbeddingError, GeminiEmbeddingModelListQuery}, LlmClient, RequestOptions};

async fn discover_gemini_models(
    client: &LlmClient,
    options: &RequestOptions,
) -> Result<(), EmbeddingError> {
    let first = client
        .embeddings()
        .list_gemini_models("gemini", &GeminiEmbeddingModelListQuery::default(), options)
        .await?;

    if let Some(resource) = first.models.first() {
        let detail = client
            .embeddings()
            .get_gemini_model("gemini", &resource.resource_name, options)
            .await?;
        assert_eq!(detail.id, resource.id);
    }

    if let Some(token) = first.next_page_token {
        let next = client
            .embeddings()
            .list_gemini_models(
                "gemini",
                &GeminiEmbeddingModelListQuery {
                    page_token: Some(token),
                    ..Default::default()
                },
                options,
            )
            .await?;
        let _ = next.models;
    }
    Ok(())
}
```

References: [Google Gemini embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings), [Gemini embeddings REST API](https://ai.google.dev/api/embeddings), [Gemini Models API](https://ai.google.dev/api/models).
