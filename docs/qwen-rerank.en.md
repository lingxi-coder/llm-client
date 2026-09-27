# Qwen Model Studio Text Reranking

[简体中文](qwen-rerank.md)

`QwenRerankService` wraps Alibaba Cloud Model Studio's provider-native `qwen3.7-text-rerank` HTTP API. A call submits one query and its candidate documents, then returns provider-ranked original document indices, relevance scores, and optional token usage. This is a different contract from the OpenAI-compatible `qwen3-rerank` `/compatible-api/v1/reranks` route; this module implements only the native API.

The current official [Text Rerank API](https://help.aliyun.com/en/model-studio/text-rerank-api) lists this exact native HTTP route:

```text
POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
```

The request uses a Bearer Model Studio API key and the `model`, `input.query`, `input.documents`, and optional `parameters.top_n` / `parameters.instruct` fields. Successful results are in `output.results`; each `index` addresses the original input `documents`, and `relevance_score` ranges from 0 to 1. The service preserves the provider's order and original indices.

Callers provide an explicit profile, stable non-secret account identifier, region, and workspace ID, then pass the same account identifier and that region's API key through `RequestOptions`. The workspace ID becomes part of the request hostname, so it must be one DNS label without a path. The current native HTTP reference gives an exact route only for Beijing. Although it has general regional URL guidance, it does not list a complete URL for this model and route in another region; this implementation does not infer a Singapore or other regional address.

```rust,ignore
use lingxi_llm_client::{
    qwen_rerank::{QwenRerankRegion, QwenRerankRequest, QwenRerankScope, QwenRerankService},
    RequestOptions,
};

let scope = QwenRerankScope::new(
    "qwen-beijing",
    "tenant-42",
    QwenRerankRegion::Beijing,
    "llm-your-workspace-id",
)?;
let service = QwenRerankService::new(transport, scope)?;
let request = QwenRerankRequest::new(
    "What is the return window?",
    [
        "Returns are accepted within 30 days.",
        "Our office is in London.",
    ],
)
.with_top_n(1)
.with_instruction("Retrieve passages that answer the query.");
let result = service.rerank(&request, &RequestOptions {
    credential: Some(api_key),
    account_scope: Some("tenant-42".into()),
    ..Default::default()
}).await?;
for hit in result.results {
    println!("input document {} scored {}", hit.index, hit.relevance_score);
}
```

Before sending, the service validates non-empty query/documents, a maximum of 500 documents, `top_n`, and account scope. Model Studio also documents a 30,000-token maximum per query/document and recommends no more than 120,000 input tokens per request. This crate does not estimate the provider tokenizer, so Model Studio rejects token-limit violations. When `top_n` exceeds the document count, Model Studio returns all documents.

Each call sends exactly one HTTP request and is never retried automatically. A transport interruption or server 5xx is classified as `Unknown`; HTTP 4xx is `Rejected`; and a malformed response after HTTP 2xx is `Accepted`. These outcomes are not instructions to replay the request. Contract tests use a mock transport and do not access or bill a real account.
