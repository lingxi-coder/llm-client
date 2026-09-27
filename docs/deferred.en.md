# xAI Deferred Chat

[简体中文](deferred.md)

`client.deferred()` wraps single submission and retrieval for xAI [Deferred Chat Completions](https://docs.x.ai/developers/advanced-api-usage/deferred-chat-completions). The built-in `grok` Chat profile has a separate Deferred route; `grok-responses` does not use this Chat-only API. Configuration v3 can inherit or explicitly disable the route. The service does not poll, retry, or fail over.

Submit a `ChatRequest` with a stable, non-secret account identifier in `RequestOptions.account_scope`. The client encodes with the selected Chat codec, adds `deferred: true`, and returns a `DeferredJobRef` scoped to provider, profile, submission endpoint, result endpoint, account, and model. Responses continuation and hosted tools are not supported in this combination. Unresolved application attachments fail before sending. The host remains responsible for executing ordinary function tools.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::ChatRequest;
use lingxi_llm_client::deferred::{DeferredError, DeferredJob, DeferredJobRef, DeferredPoll};

async fn submit(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<DeferredJob, DeferredError>
{
    client.deferred().submit("grok", request, options).await
}

async fn check(client: &LlmClient, ticket: DeferredJobRef, options: &RequestOptions)
    -> Result<DeferredPoll, DeferredError>
{
    client.deferred().fetch_once(ticket, options).await
}
```

`fetch_once()` **consumes** its ticket. HTTP `202` returns `DeferredPoll::Pending(ticket)` for a later caller-managed check; HTTP `200` returns `Completed` with a decoded `ChatResponse` and full native JSON, without a reusable ticket. A catalog change during the queue does not prevent retrieval: the ticket retains the original wire model ID. xAI allows a completed result to be retrieved only once within 24 hours, so persist the data you need as soon as it is returned. A submission disconnect yields `SubmitOutcomeUnknown`; an accepted submission without a valid ID yields `SubmitAcceptedUnknownId`. Neither is safe to blindly resubmit. A fetch disconnect yields `FetchOutcomeUnknown` with the ticket, but the server may already have consumed the result, so the client does not retry. An expired or consumed result retains the provider's HTTP status and body rather than being treated as pending; decode failures are separate.

Tests use mocks; no live account acceptance was run. [xAI REST reference](https://docs.x.ai/developers/rest-api-reference/inference/chat-completions).
