# OpenAI Responses background jobs

[简体中文](background.md)

`client.background()` implements OpenAI [Background mode](https://developers.openai.com/api/docs/guides/background). The built-in `openai` profile has an independent Responses background route, which configuration v3 can inherit or disable. Submission uses the selected model's Responses encoder and validation, then adds `background: true`. The API supports non-streaming submission, one-shot retrieval and cancellation, plus a resumable native event stream and a provider-neutral decoded stream. The host owns polling intervals, reference persistence, and tool execution.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::ChatRequest;
use lingxi_llm_client::background::{BackgroundError, BackgroundJob, BackgroundJobRef};

async fn submit(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<BackgroundJob, BackgroundError>
{
    client.background().submit("openai", request, options).await
}

async fn check(client: &LlmClient, reference: &BackgroundJobRef, options: &RequestOptions)
    -> Result<BackgroundJob, BackgroundError>
{
    client.background().get(reference, options).await
}
```

Streaming submission uses `background=true, stream=true`. Each `submit_stream().next_event()` returns native JSON and a cursor. Persist the cursor after processing the event, then pass it to `resume_stream()` after a disconnect. The recovery GET sends `stream=true&starting_after=<sequence_number>` without resubmitting the job. There is no usable cursor until an event supplies the response ID; a disconnect before then leaves the submission outcome unknown. Stream errors retain the last delivered cursor, and a terminal event ends the stream.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::ChatRequest;
use lingxi_llm_client::background::{BackgroundError, BackgroundEventCursor, BackgroundStreamError};

async fn stream_job(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<Option<BackgroundEventCursor>, BackgroundError>
{
    let mut stream = client.background().submit_stream("openai", request, options).await?;
    let mut last = None;
    loop {
        match stream.next_event().await {
            Ok(Some(event)) => last = Some(event.cursor),
            Ok(None) => break,
            Err(BackgroundStreamError::Interrupted { cursor, .. }) => {
                // Persist cursor, then reconnect with client.background().resume_stream(cursor, options).
                last = cursor.map(|cursor| *cursor);
                break;
            }
            Err(_) => break,
        }
    }
    Ok(last)
}
```

For provider-neutral output, use `submit_chat_stream()` and `resume_chat_stream()`. Each event retains the native JSON and cursor, alongside decoded `StreamEvent`s. The terminal `response.completed` or `response.incomplete` event also includes the reconstructed `ChatResponse`. A `response.failed` event remains an error and carries its native event and cursor. Earlier deltas have already been yielded and remain caller-owned. A resumed decoder starts after the cursor; retain prior yielded deltas yourself if you need a continuous event history. The terminal event contains the full response and is decoded independently, so the final `ChatResponse` is complete after resume.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::{ChatRequest, ChatResponse};

async fn read_response(
    client: &LlmClient,
    request: &ChatRequest,
    options: &RequestOptions,
) -> Result<Option<ChatResponse>, Box<dyn std::error::Error>> {
    let mut stream = client.background().submit_chat_stream("openai", request, options).await?;
    while let Some(event) = stream.next_event().await? {
        // Persist event.cursor after processing it. Preserve event.native when
        // the provider-specific event data is needed by the host.
        if let Some(response) = event.response {
            return Ok(Some(response));
        }
    }
    Ok(None)
}
```

Supply a stable, non-secret account identity in `RequestOptions.account_scope`. `BackgroundJobRef` binds the provider, profile, endpoint, account, model, and response ID; retrieval and cancellation check that scope before sending. `queued` and `in_progress` are pending; `completed`, `failed`, `cancelled`, and `incomplete` are terminal. Completed and incomplete results include a decoded `ChatResponse` when valid; a completed result also carries an account-bound continuation reference. Every status retains the full native JSON. A failed response retains its native error rather than being presented as successful text.

A disconnected submission returns `SubmitOutcomeUnknown` and must not be blindly resubmitted. A disconnected cancellation returns `CancelOutcomeUnknown` with the reference; retrieve status to reconcile it. The client never retries, polls, or fails over automatically. OpenAI documents roughly ten minutes of temporary polling retention unless background results are explicitly stored, so the host should fetch and persist needed results promptly. No live account acceptance has been run. See [background mode and stream recovery](https://developers.openai.com/api/docs/guides/background), [retrieve](https://developers.openai.com/api/reference/typescript/resources/responses/methods/retrieve), and [cancel](https://developers.openai.com/api/reference/go/resources/beta/subresources/responses/methods/cancel).

## Deleting a stored Response

`background().delete(&reference, &options)` sends one `DELETE /responses/{id}` and returns `BackgroundDeletion` only after matching `id`, `object: "response"`, and `deleted: true`. The receipt preserves the scoped reference and native fields. Deletion is separate from cancelling inference; the host decides when to remove retained results. It checks the same provider/profile/endpoint/account scope as retrieval, and does not require the model to remain in the current catalog.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions, background::BackgroundJobRef};
async fn remove_saved_response(
    client: &LlmClient,
    reference: &BackgroundJobRef,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let deleted = client.background().delete(reference, options).await?;
    assert_eq!(deleted.reference.response_id, reference.response_id);
    Ok(())
}
```

Transport/body failures, server errors, and malformed success receipts return `DeleteOutcomeUnknown` with the reference and underlying error. The client never retries or infers success from a `404`. Reconcile retained state before deciding what to do next. [Official deletion reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/delete).
