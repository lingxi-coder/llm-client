# Kimi Batch

`kimi_batch` provides typed submit, query, one-page list, cancel, and result-stream operations for Kimi's Batch API. The current official guide documents the international API host `https://api.moonshot.ai/v1`, the `/v1/chat/completions` endpoint, and the `kimi-k2.6` and `kimi-k2.7-code` models. It demonstrates a `24h` completion window. The module does not accept arbitrary URLs, model IDs, or endpoints.

Upload the input JSONL file through the Kimi Files API with `purpose="batch"` before submitting it. This module starts from that uploaded file ID; it does not construct or upload JSONL files. Each input row must be a POST to `/v1/chat/completions`, use one supported model, and have a unique `custom_id`. A file must use a single model, be non-empty, have a `.jsonl` extension, and be no larger than 100 MB. Kimi currently disallows parameters such as `temperature`, `top_p`, `n`, `presence_penalty`, and `frequency_penalty` in these Batch requests.

```rust,no_run
use lingxi_llm_client::{
    providers::kimi::batch::{
        KimiBatchInputFileRef, KimiBatchListOptions, KimiBatchModel, KimiBatchScope,
        KimiBatchService, KimiBatchSubmitOptions,
    },
    protocol::Secret,
    transport::HttpTransport,
};

# async fn example(api_key: String, uploaded_file_id: String) -> Result<(), Box<dyn std::error::Error>> {
let http = HttpTransport::new()?;
let scope = KimiBatchScope::new(
    "kimi-intl-production",
    "account-123",
    "https://api.moonshot.ai/v1",
)?;
let request_options = lingxi_llm_client::RequestOptions {
    credential: Some(Secret::new(api_key)),
    ..Default::default()
};
let service = KimiBatchService::new(&http, scope.clone())?;
let input = KimiBatchInputFileRef::new(
    scope,
    KimiBatchModel::KimiK2_6,
    uploaded_file_id,
)?;

let submitted = service
    .submit(&input, KimiBatchSubmitOptions::default(), &request_options)
    .await?;
let _page = service.list(&KimiBatchListOptions::new().limit(10), &request_options).await?;
let latest = service.get(&submitted.reference, &request_options).await?;
if let Some(output) = latest.output_ref() {
    let mut chunks = service.stream_result(&output, &request_options).await?;
    use futures::StreamExt;
    while let Some(chunk) = chunks.next().await {
        // Consume or persist each raw JSONL byte chunk as it arrives.
        let _chunk = chunk?;
    }
}
# Ok(())
# }
```

Job and result references retain the provider, profile, endpoint fingerprint, account scope, and task or file IDs. A reference from another profile, endpoint, or account is rejected before a request is sent. Credentials are not stored in references. If submission or cancellation loses its transport response, the service reports the outcome as unknown and does not retry a potentially accepted operation. Use a known job reference to query the provider when possible. Large JSONL results are returned as a byte stream; the service does not parse or buffer the full file.

The current Kimi Batch documentation does not specify a mainland-China Batch endpoint, so this service accepts only the international endpoint above. It does not poll automatically; callers decide when to query until a terminal status. The output reference becomes available after the provider reports `completed`. The snapshot also exposes `error_file_id` for caller-managed error handling.

`list` fetches one page using the documented `after` cursor and `limit` parameter (default 20); pass the final item's `batch_id` as `after` when `has_more` is true. Kimi's list response does not include the model, so list items retain the scoped task identity and native response without assigning a model. Call `reference_with_model` only when the caller can identify the model for that task.

See the official [Batch API guide](https://platform.kimi.ai/docs/guide/use-batch-api), [Create Batch API](https://platform.kimi.ai/docs/api/batch-create), [List Batches API](https://platform.kimi.ai/docs/api/batch-list), [Retrieve Batch API](https://platform.kimi.ai/docs/api/batch-retrieve), and [Cancel Batch API](https://platform.kimi.ai/docs/api/batch-cancel).
