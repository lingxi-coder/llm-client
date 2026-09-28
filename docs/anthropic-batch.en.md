# Anthropic Messages Batch

`AnthropicBatchService` exposes Anthropic's Messages Batch lifecycle as an independent provider service. It sends the documented inline `requests` array, retrieves and lists batch state, requests cancellation, deletes completed batches, and parses result records from the results response as they arrive. It does not write a JSONL file or buffer all results in memory.

The bound service created by `AnthropicClient::batch(scope)` uses the profile's registered authenticator and the per-operation credential. A directly constructed standalone service continues to use an Anthropic API key.

```rust,no_run
use lingxi_llm_client::{
    providers::anthropic::batch::{
        AnthropicBatchInput, AnthropicBatchMessage, AnthropicBatchParams,
        AnthropicBatchRequest, AnthropicBatchRole, AnthropicBatchScope,
        AnthropicBatchService,
    },
    protocol::Secret,
    Transport,
};
use serde_json::json;

async fn use_batch(
    transport: &dyn Transport,
    api_key: String,
) -> Result<(), Box<dyn std::error::Error>> {
let scope = AnthropicBatchScope::new(
    "anthropic-production",
    "account-42",
    "https://api.anthropic.com",
    Some("wrkspc_example".into()),
)?;
let params = AnthropicBatchParams::new(
    "claude-sonnet-5",
    1024,
    vec![AnthropicBatchMessage {
        role: AnthropicBatchRole::User,
        content: json!("Summarize this document."),
    }],
)?
.with_parameter("system", json!("Be concise"))?;
let input = AnthropicBatchInput::new(vec![AnthropicBatchRequest::new("document-1", params)?])?;

// `transport` implements the crate's Transport trait. Keep the API key in
// the host's credential store and pass its Secret value through each operation's RequestOptions.
let request_options = lingxi_llm_client::RequestOptions {
    credential: Some(Secret::new(api_key)),
    ..Default::default()
};
let batches = AnthropicBatchService::new(transport, scope)?;
let created = batches.create(&input, &request_options).await?;
let current = batches.get(&created.reference, &request_options).await?;
let page = batches.list(&Default::default(), &request_options).await?;
let results = batches.stream_results(&current.reference, &request_options).await?;
let _ = (page, results);
Ok(())
}
```

Each `custom_id` must be unique in a batch and match `[a-zA-Z0-9_-]{1,64}`. The typed core requires a model, at least one message, and `max_tokens > 0`; use `with_parameter` for other Messages fields such as `system`, `tools`, and `thinking`. `stream: true` and `speed` are rejected because Anthropic does not support them in Batch requests. Current limits are 100,000 requests and 256 MiB per create request.

Each `create`, `get`, `list`, `cancel`, and `delete` call sends one HTTP request. The caller controls polling and recovery. Deletion is only available after processing ends; cancel an in-progress batch and confirm its terminal state with `get` before calling `delete`. A transport or response-read failure after a create, cancel, or delete request is reported as an unknown outcome, and the service does not retry it. A successful HTTP response with an unreadable mutation body is also marked unknown; a definite non-2xx response is a provider error.

`AnthropicBatchRef` binds provider, profile, endpoint fingerprint, and account/workspace scope. A service rejects references from another scope before making a request. The response's `results_url` is retained for inspection, but routing uses the endpoint bound in the reference and the batch ID rather than following a response-provided URL.

`stream_results` yields one `AnthropicBatchItemResult` per JSONL record. Output order is not guaranteed, so match by `custom_id`. Success, item error, cancellation, expiration, and future result types have distinct typed outcomes. Item errors remain stream values and do not hide later results. Transport errors and malformed or over-16-MiB lines end the stream with an error.

The caller owns polling cadence, result persistence, and any resubmission decisions. Anthropic documents processing windows up to 24 hours and result availability for 29 days from creation. The service does not auto-poll, auto-retry, or make live calls during construction.

References: [Anthropic Batch processing guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing), [Create a Message Batch](https://platform.claude.com/docs/en/api/messages/batches/create), and [Delete a Message Batch](https://platform.claude.com/docs/en/api/http/messages/batches/delete).
