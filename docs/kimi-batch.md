# Kimi Batch

`kimi_batch` provides typed submit, query, one-page list, cancel, and result-stream operations for the Kimi API Batch service. The current official guide documents the international API host `https://api.moonshot.ai/v1`, the `/v1/chat/completions` endpoint, and the models `kimi-k2.6` and `kimi-k2.7-code`. It demonstrates a `24h` completion window. The module does not accept arbitrary URLs or other model and endpoint strings.

The input JSONL file must first be uploaded through the Kimi Files API with `purpose="batch"`. This module starts from that uploaded file ID; it does not build or upload the JSONL file. Each input row must be a POST to `/v1/chat/completions`, use one of the supported models, and have a unique `custom_id`. The whole file must use the same model, be non-empty, have a `.jsonl` extension, and be at most 100 MB. Kimi currently disallows parameters including `temperature`, `top_p`, `n`, `presence_penalty`, and `frequency_penalty` in these Batch requests.

```rust,no_run
use lingxi_llm_client::{
    kimi_batch::{
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
let service = KimiBatchService::new(&http, Secret::new(api_key), scope.clone())?;
let input = KimiBatchInputFileRef::new(
    scope,
    KimiBatchModel::KimiK2_6,
    uploaded_file_id,
)?;

let submitted = service
    .submit(&input, KimiBatchSubmitOptions::default())
    .await?;
let _page = service.list(&KimiBatchListOptions::new().limit(10)).await?;
let latest = service.get(&submitted.reference).await?;
if let Some(output) = latest.output_ref() {
    let mut chunks = service.stream_result(&output).await?;
    use futures::StreamExt;
    while let Some(chunk) = chunks.next().await {
        // Consume or persist each raw JSONL byte chunk as it arrives.
        let _chunk = chunk?;
    }
}
# Ok(())
# }
```

Job and result references retain the provider, profile, endpoint fingerprint, account scope, and task/file identifiers. A reference from another profile, endpoint, or account is rejected before a request is sent. Credentials are not stored in references. If submission or cancellation loses its transport response, the service reports the outcome as unknown; it does not retry a possibly accepted operation. Callers should query using a known job reference when one is available. Large result JSONL files are exposed as a byte stream, and are not parsed or buffered by the service.

The current Kimi Batch documentation does not specify a mainland-China Batch endpoint. This service therefore accepts only the international endpoint above. It also does not automatically poll: callers choose when to query until a terminal status. Output references become available once the provider reports `completed`; `error_file_id` remains available on the snapshot for caller-managed error handling.

`list` fetches one page with the documented `after` cursor and `limit` parameter (default 20). If `has_more` is true, pass the final item's `batch_id` as `after` in another explicit call. Kimi's list response omits the model, so list items retain scoped task identity and the native response without assigning one. Call `reference_with_model` only when the caller knows which model belongs to that task.

See the official [Batch API guide](https://platform.kimi.ai/docs/guide/use-batch-api), [Create Batch API](https://platform.kimi.ai/docs/api/batch-create), [List Batches API](https://platform.kimi.ai/docs/api/batch-list), [Retrieve Batch API](https://platform.kimi.ai/docs/api/batch-retrieve), and [Cancel Batch API](https://platform.kimi.ai/docs/api/batch-cancel).
