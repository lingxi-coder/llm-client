# Gemini Batch API

`gemini_batch` exposes Gemini Developer API `generateContent` Batch operations for inline requests and JSONL files: create, update, get, one-page list, cancel, delete, and inline or file-backed results. A separate typed lifecycle supports asynchronous `EmbedContent` batches, typed inline item results, and incremental typed JSONL result rows. File input uses the documented resumable Gemini Files API upload. The service stores only the transport and non-secret scope; the host passes `&Secret<String>` to every operation and owns credential refresh and rotation. The service does not automatically poll, paginate, or retry. Generation JSONL output remains raw bytes; embedding JSONL output is decoded one row at a time.

Routes and JSON shapes follow Google's [Batch API guide](https://ai.google.dev/gemini-api/docs/batch-api), [Batch API REST reference](https://ai.google.dev/api/batch-api), and [Files API reference](https://ai.google.dev/api/files): create with `POST /v1beta/models/{model}:batchGenerateContent`; update with `PATCH /v1beta/batches/{batchId}:updateGenerateContentBatch`; get with `GET /v1beta/batches/{batchId}`; list with `GET /v1beta/batches?pageSize=...&pageToken=...`; cancel with `POST /v1beta/batches/{batchId}:cancel`; and delete with `DELETE /v1beta/batches/{batchId}`. The API key is sent in `x-goog-api-key`.

```rust,no_run
use lingxi_llm_client::{
    gemini_batch::{
        GeminiBatchCreateRequest, GeminiBatchError, GeminiBatchGenerateContentRequest,
        GeminiBatchInput, GeminiBatchListOptions, GeminiBatchRequest, GeminiBatchScope,
        GeminiBatchService,
    },
    protocol::Secret,
    transport::Transport,
};
use serde_json::json;

async fn run_batch(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;

    let request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "Summarize this text."}]
    })])?;
    let row = GeminiBatchRequest::new(request).with_key("summary-1")?;
    let input = GeminiBatchInput::new(vec![row])?;
    let create = GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "summaries-september",
        input,
    )?;

    // Creation enqueues work and is non-idempotent. Persist this scoped ref.
    let created = service.create(&create, &api_key).await?;

    // The caller controls when to check status; each call makes one request.
    let current = service.get(&created.reference, &api_key).await?;
    let _page = service.list(&GeminiBatchListOptions::new(), &api_key).await?;
    let results = service.results(&current.reference, &api_key).await?;
    if let Some(items) = results.items {
        for item in items {
            if let Some(response) = item.response {
                let _native_generate_content_response = response;
            } else if let Some(error) = item.error {
                let _per_item_error = error;
            }
        }
    }

    // Cancellation is best effort. Call get later to confirm the final state.
    // service.cancel(&created.reference, &api_key).await?;
    Ok(())
}
```

Inline input uses the REST schema's `batch.input_config.requests.requests[]` shape. `GeminiBatchGenerateContentRequest::from_value` verifies that a request is an object with a non-empty `contents` array and retains other GenerateContent fields. The complete serialized create body must be smaller than 20,000,000 bytes.

For file input, each JSONL line has a unique caller-supplied `key` and one native REST `request` object. `GeminiBatchJsonlInput` validates the rows and encodes one JSON object per line; `upload_input_jsonl` streams those encoded rows through the Files API resumable protocol and returns a scoped `GeminiBatchFileRef`. To stream an existing local file without buffering it, pass its exact byte length and a one-shot stream to `upload_input_stream`. Both upload methods enforce Google's 2 GB input limit, and neither retries an uncertain upload. Pass the returned file ref to `GeminiBatchInput::from_file`; create then sends `batch.input_config.file_name`.

```rust,no_run
use futures::StreamExt;
use lingxi_llm_client::gemini_batch::{
    GeminiBatchCreateRequest, GeminiBatchError, GeminiBatchFileRef,
    GeminiBatchGenerateContentRequest, GeminiBatchInput, GeminiBatchJsonlInput,
    GeminiBatchJsonlRequest, GeminiBatchScope, GeminiBatchService, GeminiBatchState,
};
use lingxi_llm_client::{protocol::Secret, transport::Transport};
use serde_json::json;

async fn run_file_batch(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;
    let request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "Summarize this text."}]
    })])?;
    let input = GeminiBatchJsonlInput::new(vec![
        GeminiBatchJsonlRequest::new("summary-1", request)?,
    ])?;
    let uploaded = service
        .upload_input_jsonl("summaries.jsonl", input, &api_key)
        .await?;
    let create = GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "summaries-september",
        GeminiBatchInput::from_file(uploaded),
    )?;
    let created = service.create(&create, &api_key).await?;

    // Read status explicitly. A production caller can persist and reuse this ref.
    let current = service.get(&created.reference, &api_key).await?;
    if current.state == Some(GeminiBatchState::Succeeded) {
        if let Some(output_file) = current.output_file {
            let mut output = service.download_results(&output_file, &api_key).await?;
            while let Some(chunk) = output.next().await {
                let _jsonl_bytes = chunk?;
            }
        }
    }

    // A resource name returned by another upload path can also be bound locally.
    let _same_scope_file = GeminiBatchFileRef::from_resource_name(
        service.scope(),
        "files/preuploaded-123",
    )?;
    Ok(())
}
```

Google returns inline output in `operation.response.output.inlinedResponses.inlinedResponses`. Each item includes `metadata` and either a `response` or an `error`. Results remain provider-native JSON, and an item error is not promoted to a batch-level request failure. If the operation does not contain inline output yet, `GeminiBatchResults::items` is `None`; when output is present, it is `Some` with every successful or failed item.

For file-backed batches, the operation response can instead contain `response.output.responsesFile`. The decoded file resource is available as `GeminiBatchSnapshot::output_file`; `download_results` returns its JSONL bytes as a stream. Each output line is a provider-native response or status object. The caller controls when to read status and download results; this module does not poll or buffer the file.

Batch references are bound to the `google` provider ID, profile, API endpoint fingerprint, and caller-supplied non-secret `account_scope`. A reference used outside any of those scopes is rejected before a request is sent. Persist the scope identity together with the reference.

Batch creation, update, deletion, and file upload are not automatically retried. A transport error during upload, create, update, cancel, or delete returns `OutcomeUnknown`; a successful HTTP status with an undecodable result returns `OutcomeUnknownResponse`. Google documents delete as removing client interest in the operation result; it does not cancel processing. The cancellation endpoint is best effort: a successful return means the cancellation request was accepted, and the caller should make a separate later status read to confirm the final state.

`update_generate_content_batch` sends the documented `PATCH /v1beta/batches/{batchId}:updateGenerateContentBatch` request. `GeminiBatchUpdateRequest` encodes the required resource fields `model`, `displayName`, and `inputConfig`; optional `priority` is an int64 serialized as a decimal string, and negative priorities are documented as valid. `updateMask` is optional and accepts the resource field names `model`, `displayName`, `inputConfig`, and `priority`. For file input, the resource body uses `inputConfig.fileName`. Inline update resources use the same client-side 20,000,000-byte cap as inline create requests; use a file input for larger batches. Google returns a `GenerateContentBatch` resource directly, so the client checks its name against the scoped reference and decodes its required fields. Transport errors or malformed 2xx responses return `OutcomeUnknown` or `OutcomeUnknownResponse` with the reference; no retry occurs. The REST reference documents no additional priority range or batch-state precondition, so the client adds none.

```rust,no_run
use lingxi_llm_client::gemini_batch::{
    GeminiBatchError, GeminiBatchGenerateContentRequest, GeminiBatchInput,
    GeminiBatchRequest, GeminiBatchSnapshot, GeminiBatchUpdateField,
    GeminiBatchUpdateRequest, GeminiBatchService,
};
use lingxi_llm_client::protocol::Secret;
use serde_json::json;

async fn update_example(
    service: &GeminiBatchService<'_>,
    created: &GeminiBatchSnapshot,
    api_key: &Secret<String>,
) -> Result<(), GeminiBatchError> {
    let replacement_input = GeminiBatchInput::new(vec![GeminiBatchRequest::new(
        GeminiBatchGenerateContentRequest::new(vec![json!({
            "role": "user",
            "parts": [{"text": "updated prompt"}]
        })])?,
    )
    .with_key("updated-row")?])?;
    let update = GeminiBatchUpdateRequest::new(
        "gemini-3.8-flash",
        "renamed-batch",
        replacement_input,
    )?
    .with_priority(-5)
    .with_update_mask([
        GeminiBatchUpdateField::DisplayName,
        GeminiBatchUpdateField::Priority,
    ])?;
    let _updated = service
        .update_generate_content_batch(&created.reference, &update, api_key)
        .await?;
    Ok(())
}
```

## Embedding batches

The same service exposes the separate asynchronous `EmbedContent` Batch operation through typed APIs. It posts to `POST /v1beta/models/{model}:asyncBatchEmbedContent`, placing the model in each native `EmbedContentRequest` and keeping current embedding options inside `embedContentConfig`. The deprecated top-level `taskType`, `title`, and `outputDimensionality` fields are not emitted. Google documents 128–3072 output dimensions for `gemini-embedding-001` and `gemini-embedding-2`; `gemini-embedding-2` does not accept `taskType`, and a title is accepted only for `RETRIEVAL_DOCUMENT`.

```rust,no_run
use lingxi_llm_client::{
    gemini_batch::{
        GeminiBatchEmbeddingConfig, GeminiBatchEmbedContentItem,
        GeminiBatchEmbedContentRequest, GeminiBatchListOptions,
        GeminiEmbeddingBatchCreateRequest, GeminiEmbeddingBatchInput,
        GeminiBatchError, GeminiBatchScope, GeminiBatchService,
    },
    protocol::Secret,
    transport::Transport,
};

async fn embed_corpus(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;
    let config = GeminiBatchEmbeddingConfig::new()
        .with_output_dimensionality(768)?;
    let request = GeminiBatchEmbedContentRequest::text("Batch embeddings are useful.")?
        .with_config(config);
    let input = GeminiEmbeddingBatchInput::new(vec![
        GeminiBatchEmbedContentItem::new(request).with_key("row-1")?,
    ])?;
    let create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "corpus-september",
        input,
    )?;

    let created = service.create_embedding_batch(&create, &api_key).await?;
    let current = service
        .get_embedding_batch(&created.reference, &api_key)
        .await?;
    let _page = service
        .list_embedding_batches(&GeminiBatchListOptions::new(), &api_key)
        .await?;
    let results = service
        .embedding_batch_results(&current.reference, &api_key)
        .await?;
    if let Some(items) = results.items {
        for item in items {
            if let Some(response) = item.response {
                // First-party responses can omit values and report tensor shape.
                let _shape = response.embedding.shape;
            } else if let Some(error) = item.error {
                let _per_request_error = error;
            }
        }
    }
    Ok(())
}
```

For file input, build keyed native `EmbedContentRequest` rows with `GeminiEmbeddingBatchJsonlInput`, then call `upload_embedding_input_jsonl`. The encoder validates unique, non-empty keys and the documented 2 GB file limit before starting upload. The returned `GeminiEmbeddingBatchFileRef` carries the model, request keys in input order, and output dimensions; pass it through `GeminiEmbeddingBatchInput::from_file` to create the job. `upload_embedding_input_stream` remains available when the host already has a JSONL stream.

```rust,no_run
use lingxi_llm_client::gemini_batch::{
    GeminiBatchEmbeddingConfig, GeminiBatchEmbedContentRequest,
    GeminiEmbeddingBatchCreateRequest, GeminiEmbeddingBatchInput,
    GeminiEmbeddingBatchJsonlInput, GeminiEmbeddingBatchJsonlRequest,
    GeminiBatchError, GeminiBatchScope, GeminiBatchService,
};
use lingxi_llm_client::{protocol::Secret, transport::Transport};

async fn embed_from_file(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;
    let input = GeminiEmbeddingBatchJsonlInput::new(
        "gemini-embedding-2",
        vec![GeminiEmbeddingBatchJsonlRequest::new(
            "row-1",
            GeminiBatchEmbedContentRequest::text("One document.")?
                .with_config(GeminiBatchEmbeddingConfig::new()
                    .with_output_dimensionality(768)?),
        )?],
    )?;
    let file = service
        .upload_embedding_input_jsonl("embeddings.jsonl", input, &api_key)
        .await?;
    let create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "file-backed-embeddings",
        GeminiEmbeddingBatchInput::from_file(file),
    )?;
    let created = service.create_embedding_batch(&create, &api_key).await?;
    let current = service.get_embedding_batch(&created.reference, &api_key).await?;
    if let Some(file) = current.output_file {
        let mut rows = service.download_embedding_results(&file, &api_key).await?;
        use futures::StreamExt;
        while let Some(row) = rows.next().await {
            let _typed_embedding_or_item_error = row?;
        }
    }
    Ok(())
}
```

Embedding references serialize an explicit operation discriminator and cannot be passed to the generateContent lifecycle. Create-time references retain model and known dimensions; list references keep model/dimension metadata unknown when Google omits it. Google documents inline and file outputs in input request order ([REST output schema](https://ai.google.dev/api/embeddings#EmbedContentBatchOutput)). File output is streamed as typed JSONL items. For typed JSONL uploads, known dimensions and keys are checked by row position, keyed rows must match that order, and keyless rows inherit the corresponding input key. The stream rejects missing or extra rows and dimension mismatches. When a caller uploads a raw file without typed request metadata, it can attach a uniform expected dimension to validate each returned vector.

`GeminiEmbeddingBatchResultStream` parses one JSONL item at a time and limits each line to 64 MiB without limiting the total file size. The caller still controls status reads and downloads; the service does not poll or paginate automatically. Cancellation remains best effort, and delete removes client interest without cancelling processing.
