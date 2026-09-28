# OpenRouter Batch

[简体中文](openrouter-batch.md)

`OpenRouterBatchService` implements OpenRouter's inline Batch API. Requests are submitted together in one JSON array to `POST /api/v1/batches`; there is no file upload, JSONL input, or separate result-file download. Batch results are included in a retrieved batch object.

The typed API covers Chat Completions, Responses, Anthropic Messages, and embeddings. Chat, Responses, and Messages bodies can include URL-only image/file parts; embedding inputs remain text-only. Each batch has one model and one endpoint. Every `custom_id` must be unique. A batch can optionally set `provider.only`; other synchronous provider-routing preferences are not sent. Request bodies deliberately expose typed fields rather than an arbitrary JSON passthrough.

```rust,no_run
use lingxi_llm_client::{
    providers::openrouter::batch::{
        OpenRouterBatchChatBody, OpenRouterBatchChatMessage,
        OpenRouterBatchChatRole, OpenRouterBatchEndpoint,
        OpenRouterBatchInput, OpenRouterBatchLine,
        OpenRouterBatchRequestBody, OpenRouterBatchScope,
    },
    protocol::Secret,
    LlmClient,
};

fn prepare(client: &LlmClient) -> Result<(), Box<dyn std::error::Error>> {
    let scope = OpenRouterBatchScope::new("openrouter-prod", "account-123")?;
    let provider = client.provider::<lingxi_llm_client::providers::openrouter::OpenRouterClient>(scope.profile_name())?;
    let batch = provider.batch(scope)?;
    let input = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![OpenRouterBatchLine {
            custom_id: "ticket-001".into(),
            body: OpenRouterBatchRequestBody::ChatCompletions(OpenRouterBatchChatBody {
                messages: vec![OpenRouterBatchChatMessage {
                    role: OpenRouterBatchChatRole::User,
                    content: "Summarize this ticket.".into(),
                }],
                temperature: None,
                top_p: None,
                max_tokens: Some(128),
                max_completion_tokens: None,
            }),
        }],
    )?;
    let _ = (batch, input);
    Ok(())
}
```

Call `submit()` once and keep its `OpenRouterBatchJobRef`. A transport interruption or an unreadable success response after acceptance returns `OpenRouterBatchError::OutcomeUnknown`; check `list()` before deciding whether a new submission is needed. `get()` performs one query and does not poll. The job and each result preserve native response fields, usage, aggregate errors, and per-item errors. A successful batch can still contain failed requests.

`delete()` invokes OpenRouter's deletion endpoint. It removes retained input and result artifacts and is accepted only after the batch reaches a terminal status. OpenRouter does not document a Batch cancellation endpoint, so this API does not expose `cancel()`. A failed deletion request can have an unknown outcome; inspect the batch list before retrying.

`list()` accepts the documented `created_after` and `created_before` filters as Unix seconds or ISO-8601 dates/date-times. When both are supplied, the earlier bound must be strictly earlier than the later bound. The service preserves the supplied text and URL-encodes it in the query string.

References bind the `openrouter` provider ID, profile name, API endpoint fingerprint, account scope, batch ID, request endpoint, and model. The caller supplies the API key and stable, non-secret account scope; this service does not store credentials. Calls use the OpenRouter API root `https://openrouter.ai/api/v1` and never retry submissions or deletions automatically.

Typed chat, Responses, and Anthropic Messages bodies support public HTTP(S)-URL image and file references where the selected provider accepts them. Batch multimodal inputs must be URL-only: data URIs, inline file bytes, provider file IDs, audio, and video are rejected. Provider and model capability still determines whether an individual request can run. A serialized request and response are each limited to 64 MiB, and an input can contain at most 50,000 requests. The provider may reject models without a batch-capable endpoint. No live account calls are part of the mock contract tests.

Use `OpenRouterBatchChatContent::Parts` with `OpenRouterBatchChatContentPart::image_url()` or `file_url()` for Chat Completions. The file helper serializes the public URL under the documented `file.file_data` field. Use the endpoint-specific `OpenRouterBatchResponsesInput::Messages` and `OpenRouterBatchResponsesContentPart` variants for Responses; and use `OpenRouterBatchMessagesContent::Parts` with `image_url()` or `document_url()` for Anthropic Messages. URLs are passed as supplied; the client does not fetch them or turn file bytes into data URIs.

When `provider.only` is pinned, preflight applies OpenRouter's documented capability table: image URLs are listed for OpenAI, Anthropic, and xAI, with DeepInfra limited to Chat Completions; file URLs are listed for OpenAI (Responses only), Anthropic, Mistral, and DeepInfra (Chat Completions only), while xAI is listed as unsupported. Every provider in `provider.only` must support the request's modalities. Without a provider pin, the request is left to OpenRouter's provider and model routing rules.

`provider.only` accepts lowercase base provider slugs and documented slash-suffixed endpoint slugs such as `deepinfra/turbo`, preserving the exact value on the wire. Multimodal preflight checks only the known base provider; OpenRouter remains authoritative for the specific variant and model's Batch availability. URL preflight rejects only obvious local/private IP literals and local domain suffixes. It does not resolve DNS or prove that a public URL is reachable.

Official references: [OpenRouter Batch API Quickstart](https://openrouter.ai/docs/batch-quickstart), [Provider Selection](https://openrouter.ai/docs/guides/routing/provider-selection), and [OpenRouter Batch API announcement](https://openrouter.ai/blog/announcements/batch-api/).
