# xAI Native Batch API

`XaiBatchService` wraps xAI's documented batch container, request-add, status, paginated list, cancel, and paginated results operations. The xAI REST flow separates batch creation from adding requests, so `create()` and `submit()` are separate calls. Each method performs one HTTP request; the service does not retry, poll, or fetch later pages automatically.

```rust,no_run
use lingxi_llm_client::{
    protocol::Secret,
    transport::Transport,
    xai_batch::{
        XaiBatchChatCompletion, XaiBatchCreateRequest, XaiBatchInput,
        XaiBatchPageOptions, XaiBatchRequest, XaiBatchRequestBody,
        XaiBatchScope, XaiBatchService,
    },
};
use serde_json::json;

async fn xai_batch_example(
    transport: &dyn Transport,
    api_key: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let scope = XaiBatchScope::new(
        "xai-production",
        "account-42",
        "https://api.x.ai/v1",
    )?;
    let batch = XaiBatchService::new(transport, Secret::new(api_key), scope)?;
    let created = batch
        .create(&XaiBatchCreateRequest::new("nightly-evaluation")?)
        .await?;
    let requests = XaiBatchInput::new(vec![XaiBatchRequest::new(
        "case-001",
        XaiBatchRequestBody::ChatGetCompletion(XaiBatchChatCompletion::new(
            "grok-4.3",
            vec![json!({"role":"user", "content":"Summarize this report."})],
        )?),
    )?])?;
    batch.submit(&created.reference, &requests).await?;

    let _status = batch.get(&created.reference).await?;
    let page = batch
        .results(&created.reference, &XaiBatchPageOptions::new().limit(100))
        .await?;
    let _result_count = page.results.len();
    Ok(())
}
```

`XaiBatchScope` binds a profile, account, and full API endpoint. The standard endpoint is `https://api.x.ai/v1`; xAI also documents the US regional endpoint `https://us.api.x.ai/v1`. A batch reference from one scope cannot be used with another profile, account, or endpoint.

For the file-based flow, upload JSONL with `FileService::upload(..., FilePurpose::Batch)`, then bind its `ProviderFileRef` using `XaiBatchInputFileRef::from_uploaded_file(&scope, &file)` and pass it to `XaiBatchCreateRequest::with_input_file`. The Files adapter accepts xAI profiles using the ordinary OpenAI Chat or Responses protocol; the Batch call itself remains the native, model-independent xAI route. The upload uses the documented multipart `file` field and omits the optional OpenAI-compatibility `purpose`. The typed reference carries the exact profile, endpoint, and account scope plus file expiry/readiness metadata; a different scope or an already expired file is rejected before Batch creation. File-based batches are sealed, so `submit()` rejects them locally.

The current xAI Files REST upload reference caps files at 50 MB, while the Batch guide advertises a separate 200 MB file-batch limit. The client conservatively enforces the Files endpoint's 50,000,000-byte cap and does not claim that 200 MB uploads work. xAI also documents up to 50,000 requests per file; this client does not download or inspect caller-owned JSONL, so line count and row validity remain provider-validated. The caller manages the uploaded file's lifetime and cleanup.

The example uses `grok-4.3`, whose model card confirms Batch support. The [Grok 4.7](https://docs.x.ai/developers/models/grok-4.7), 4.6, 4.5, and Grok Build 0.1 model cards explicitly say Batch is unavailable. These known models and verified aliases are rejected while constructing a request; unknown models remain provider-validated.

Each request uses `batch_request_id` to correlate input and results; xAI requires IDs to be unique within a batch. The service rejects empty IDs and duplicates within one add call. If callers split a batch across multiple `submit()` calls, they must also avoid reusing IDs from earlier calls. The current typed union covers the request variants shown in xAI's documentation: `chat_get_completion`, `responses`, `image_generation`, `image_edit`, `video_generation`, and `video_extension`. Additional Chat, Responses, and media parameters are retained through `with_parameter`. xAI determines the accepted ID boundary; the client does not impose a character-set rule.

`submit()` calls `POST /v1/batches/{batch_id}/requests`; the documented successful response can be empty. Each nested request is limited to xAI's documented 25 MiB payload size. To bound serialization memory, this client also caps one inline add call at 100,000 requests and 256 MiB total JSON. Those two limits belong to this client; callers can split a large job across multiple `submit()` calls.

Batch state includes `num_requests`, `num_pending`, `num_success`, `num_error`, and `num_cancelled`. xAI documents `num_pending == 0` as completion of all added requests; snapshots preserve the full provider JSON. `list()`, `list_requests()`, and `results()` each return one page. `XaiBatchPageOptions` accepts limits from 1 to 1,000; pass a response's `pagination_token` to the next call to fetch another page.

Results retain the full native response. Recognized success responses, error messages, cancellation, and pending state have distinct outcomes. Future or currently unrecognized result shapes remain `Other` and are not reported as success. Match results by `batch_request_id`. xAI says signed URLs in image and video results expire after about an hour, so the host should preserve media it needs to keep.

If transport or response reading fails after `create()`, `submit()`, or `cancel()` dispatch, the service reports `OutcomeUnknown` and never repeats the operation. Unknown add/cancel errors include the known scoped reference so callers can reconcile with `get()`, `list_requests()`, or `results()`. A definite non-2xx response is returned as a provider error.

References: [xAI Batch API guide](https://docs.x.ai/developers/advanced-api-usage/batch-api), [Batch REST API reference](https://docs.x.ai/developers/rest-api-reference/inference/batches), and [Files upload reference](https://docs.x.ai/developers/rest-api-reference/files/upload).
