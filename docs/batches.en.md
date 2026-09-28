# Batch processing

[简体中文](batches.md)

Bind `client.provider::<OpenAiClient>(profile)?` to an exact profile, then use `provider.batches()`. Each operation takes `RequestOptions`; the client retains no credential.

`provider.batches()` manages the OpenAI Batch lifecycle independently of Chat. The built-in `openai` profile has separate Batch and result-file routes; v3 configuration can inherit, replace or disable them. This adapter follows the [OpenAI Batch guide](https://developers.openai.com/api/docs/guides/batch) and [Batch API](https://developers.openai.com/api/reference/resources/batches/methods/create). Other providers expose their own Batch services; Background and deferred jobs have separate resource APIs.

Use `encode_jsonl()` to encode up to 50,000 requests, or `write_jsonl()` to write a caller-owned file and obtain its exact byte count. Every request needs a unique `custom_id`; an input cannot mix models or enable `stream`. `write_jsonl()` validates every request before its first write; discard a partial file if an I/O write fails. Upload the JSONL as a `.jsonl` file with media type `application/x-ndjson` through the same profile's `FileService::upload(..., FilePurpose::Batch)`. This convenience method still constructs multipart in memory.

For large inputs, use `FileService::upload_batch_stream(filename, media_type, size_bytes, chunks, timeout)` with a file/generator stream and its **exact** byte count. It streams OpenAI multipart, checks the actual stream length, caps input at 200 MB, and requires a nonempty `account_scope`. Custom `Transport` implementations must override `send_stream()`; the default rejects before consuming input and does not buffer as a fallback. Upload and Batch submission are separate and never retried automatically. If an upload disconnects, its server-side outcome may be unknown; reconcile through the file list before deciding whether to send it again. Chat attachment cleanup does not manage these files. OpenAI documents a default 30-day expiration for `purpose=batch` files; the caller still manages references and cleanup during the job lifecycle.

### Durable per-request file inputs

Batch request JSONL can reference files uploaded in advance: Responses `input_file.file_id`, Responses `input_image.file_id`, and Chat Completions `file` parts using PDF file IDs. The request-level files are different from the top-level JSONL file: upload them with `FilePurpose::ModelInput` (`user_data`) and keep their `ProviderFileRef`s. OpenAI's [file-input guide](https://developers.openai.com/api/docs/guides/file-inputs) documents `user_data` for model inputs and PDF-only file IDs in Chat Completions; the [image guide](https://developers.openai.com/api/docs/guides/images-vision) also documents Responses image file IDs and uses the `vision` purpose in its examples. Responses image manifests accept either documented purpose. The Batch adapter rejects `file_url` and remote image URLs because it cannot establish that they will remain available when a queued job runs. Inline data URLs remain embedded in the JSONL and do not need an attachment manifest.

Use `encode_jsonl_with_attachments()` or `write_jsonl_with_attachments()` to bind every file ID in the JSONL to an explicit account-scoped reference. The ordinary encoders reject file IDs without this manifest, and the manifest must match the JSONL exactly. Pass the same references to `submit_with_attachments()`: it checks their provider, profile, endpoint and account scope, then retrieves each file's metadata before making the one Batch creation request. It verifies the remote ID and purpose and rejects a scheduled expiry earlier than the 24-hour Batch completion window, the documented cancellation reconciliation period, and a five-minute clock/request margin. This is a preflight check, not a lock: keep these files undeleted until the Batch is terminal and any needed output has been retrieved. The [Files API](https://developers.openai.com/api/reference/typescript/resources/files/methods/create) says files without an explicit expiry remain available until someone deletes them. The client never uploads, extends, retries, or deletes these attachment files. A transport failure during Batch creation has an unknown outcome and must be reconciled through `list()`/`get()` rather than retried blindly.

```rust,no_run
use lingxi_llm_client::providers::openai::batches::{
    BatchAttachmentRef, BatchEndpoint, BatchError, BatchJob, BatchLine,
    encode_jsonl_with_attachments,
};
use lingxi_llm_client::files::ProviderFileRef;
use lingxi_llm_client::{LlmClient, RequestOptions};
use serde_json::json;

fn input_jsonl(pdf: &ProviderFileRef) -> Result<Vec<u8>, BatchError> {
    let attachment = BatchAttachmentRef::chat_completions_pdf(pdf.clone());
    encode_jsonl_with_attachments(
        BatchEndpoint::ChatCompletions,
        &[BatchLine {
            custom_id: "document-1".into(),
            body: json!({"model":"gpt-6-sol","messages":[{"role":"user","content":[
                {"type":"file","file":{"file_id":pdf.file_id.clone()}},
                {"type":"text","text":"Summarize this PDF."}
            ]}]}),
        }],
        &[attachment],
    )
}

async fn submit_with_pdf(
    client: &LlmClient,
    jsonl_file: &ProviderFileRef,
    pdf: &ProviderFileRef,
    options: &RequestOptions,
) -> Result<BatchJob, Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let attachments = [BatchAttachmentRef::chat_completions_pdf(pdf.clone())];
    provider.batches().submit_with_attachments(
        jsonl_file, BatchEndpoint::ChatCompletions, None, &attachments, options,
    ).await.map_err(Into::into)
}
```

The same manifest pattern applies to Responses files and images with `BatchAttachmentRef::responses_file()` and `BatchAttachmentRef::responses_image()`. The API cannot inspect the uploaded JSONL file itself, so the caller must submit the exact manifest used to generate that file. A caller can still delete a file outside this client after the metadata check; the service does not claim an atomic reservation or provider guarantee against that action.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::files::ProviderFileRef;
use lingxi_llm_client::providers::openai::batches::{BatchEndpoint, BatchError, BatchJob, BatchLine, encode_jsonl, write_jsonl};
use serde_json::json;

fn input_jsonl() -> Result<Vec<u8>, BatchError> {
    encode_jsonl(BatchEndpoint::Responses, &[
        BatchLine { custom_id: "item-1".into(), body: json!({"model":"gpt-6-sol","input":"hello"}) },
        BatchLine { custom_id: "item-2".into(), body: json!({"model":"gpt-6-sol","input":"goodbye"}) },
    ])
}

fn write_input_jsonl(output: &mut impl std::io::Write) -> Result<u64, BatchError> {
    write_jsonl(BatchEndpoint::Responses, &[
        BatchLine { custom_id: "item-1".into(), body: json!({"model":"gpt-6-sol","input":"hello"}) },
    ], output)
}

async fn submit_uploaded(client: &LlmClient, input: &ProviderFileRef, options: &RequestOptions) -> Result<BatchJob, Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    provider.batches().submit(input, BatchEndpoint::Responses, None, options).await.map_err(Into::into)
}

async fn consume_result(client: &LlmClient, job: &BatchJob, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>(&job.reference.profile_name)?;
    use futures::StreamExt;
    if let Some(reference) = job.output_ref() {
        let mut rows = provider.batches().stream_result(&reference, options).await?;
        while let Some(row) = rows.next().await {
            let row = row?;
            // Match row.custom_id to the original request and persist what you need.
            let _ = row;
        }
    }
    Ok(())
}
```

`submit()` requires an uploaded file with purpose `batch` and checks provider, profile, file endpoint and `RequestOptions.account_scope`. Use `submit_with_attachments()` when the JSONL references durable Files API inputs. Both methods return a scoped `BatchJobRef`; use `get()`, `list()` and `cancel()` to manage the job. `cancelling` is not terminal, and completed requests can produce partial results. Request counts, errors and usage are preserved. A transport failure during submission or cancellation yields `BatchError::OutcomeUnknown`; reconcile through `list()` or `get()` before considering another submission.

Jobs expose `output_ref()` and `error_ref()`. `stream_result()` reads JSONL rows from the configured Files route as a `futures::stream::BoxStream`, preserving each row's `custom_id`, native `response`, and `error`; result order need not match input order. It does not buffer the complete file. Each line is capped at 16 MiB, and the stream accepts at most 50,000 unique `custom_id` values. Dropping it stops the response read. `read_result()` remains a convenience method that collects a file up to 64 MiB. Both methods check account, Batch route, and result-file route scope, and neither retries an interrupted read automatically. The service neither polls nor deletes input or result files. Tests use mocks; no live account acceptance was run.
