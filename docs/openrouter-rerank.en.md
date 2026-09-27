# OpenRouter Native Text Reranking

[简体中文](openrouter-rerank.md)

`OpenRouterRerankService` is a synchronous text wrapper for OpenRouter's native Rerank REST API. Each request sends a model slug, query, candidate text documents, and optional `top_n`. The response preserves OpenRouter's order, original input indices, scores, ID, actual model/provider, usage, and complete native JSON.

OpenRouter's [RAG guide](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/rag) uses `POST https://openrouter.ai/api/v1/rerank`, with `model`, `query`, `documents`, and `top_n` in the request body and results in `results`. The official [Rerank API reference](https://openrouter.ai/docs/api/api-reference/rerank/submit-a-rerank-request) lists the same route and structured fields: results are relevance-sorted and include `document`, `index`, and `relevance_score`; responses may also include `id`, `provider`, and `usage`. The API key is sent as a Bearer token. The API also permits multimodal document objects; this module is limited to text strings.

Callers explicitly create a scope bound to a profile, stable non-secret account identifier, and official endpoint. Each call supplies the matching account identifier and that OpenRouter account's API key through `RequestOptions`. The scope rejects non-HTTPS URLs, non-OpenRouter hosts, and any path other than the documented rerank route, so it will not forward the key to a custom host. The profile and account label are local isolation metadata and are not sent to OpenRouter.

```rust,ignore
use lingxi_llm_client::{
    openrouter_rerank::{
        OpenRouterRerankRequest, OpenRouterRerankScope, OpenRouterRerankService,
        OPENROUTER_RERANK_ENDPOINT,
    },
    RequestOptions,
};

let scope = OpenRouterRerankScope::new(
    "openrouter-prod",
    "tenant-17",
    OPENROUTER_RERANK_ENDPOINT,
)?;
let service = OpenRouterRerankService::new(transport, scope)?;
let request = OpenRouterRerankRequest::new(
    "cohere/rerank-v3.5",
    "What is the capital of France?",
    [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
    ],
)
.with_top_n(1);
let result = service.rerank(&request, &RequestOptions {
    credential: Some(api_key),
    account_scope: Some("tenant-17".into()),
    ..Default::default()
}).await?;
for hit in result.results {
    let original_document = &request.documents()[hit.index];
    println!("{} ({})", original_document, hit.relevance_score);
}
```

Before sending, the module validates the model, query, document list, `top_n >= 1`, and exact account scope. Request and response bodies are limited to 64 MiB and 16 MiB as local resource bounds; these are not advertised OpenRouter token limits. Provider-routing preferences and image/multimodal documents are outside this text-only slice.

Each call sends one HTTP request and is never retried automatically. A transport interruption or HTTP 5xx is `Unknown`; HTTP 4xx is `Rejected`; and a response that cannot be decoded, or has invalid indices/scores after HTTP 2xx, is `Accepted`. These classifications describe the available evidence and never trigger resubmission. Contract tests use a mock transport and do not call or bill the live service.
