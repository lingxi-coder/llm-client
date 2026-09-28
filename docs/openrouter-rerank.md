# OpenRouter 原生文本重排

[English](openrouter-rerank.en.md)

`OpenRouterRerankService` 为 OpenRouter 官方 Rerank REST API 提供同步文本封装。每次请求提交一个模型 slug、query 和候选文本列表，可选 `top_n`；响应保留 OpenRouter 返回的排序、原始输入索引、分数、ID、实际模型/服务商、usage 和完整原生 JSON。

OpenRouter 的 [RAG 指南](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/rag) 使用 `POST https://openrouter.ai/api/v1/rerank`，请求正文为 `model`、`query`、`documents`、`top_n`，结果读取自 `results`。官方 [Rerank API 参考](https://openrouter.ai/docs/api/api-reference/rerank/submit-a-rerank-request)列出同一条路由和结构化字段：结果按相关性排序，包含 `document`、`index` 和 `relevance_score`，并可返回 `id`、`provider` 和 `usage`。API Key 以 Bearer token 发送。请求接口也允许多模态文档对象；本模块将范围限定为文本字符串列表。

调用方需显式创建 scope，绑定 profile、稳定且非敏感的账户标识和官方 endpoint；每次调用通过 `RequestOptions` 提供相同的账户标识与对应 OpenRouter API Key。scope 会拒绝非 HTTPS、非 OpenRouter 主机名或非标准 rerank 路径，避免把 key 转发给自定义主机。账户标识和 profile 只用于本地隔离，不会发给 OpenRouter。

```rust,ignore
use lingxi_llm_client::{
    providers::openrouter::rerank::{
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

本模块在发送前检查 model、query、文档列表和 `top_n >= 1`，并要求账户标识与 scope 精确一致。请求和响应分别限制在 64 MiB 与 16 MiB，属于本地资源保护上限，不代表 OpenRouter 宣布的模型 token 限额。Provider 路由偏好以及图像/多模态文档目前不在这个文本切片的类型契约内。

每次调用只发送一个 HTTP 请求，不自动重试。传输中断或 HTTP 5xx 归类为 `Unknown`；HTTP 4xx 归类为 `Rejected`；HTTP 2xx 后响应无法解码或索引/分数无效则归类为 `Accepted`。这些分类只描述已有证据，不会自动再次提交。契约测试使用 mock transport，不调用或计费真实服务。
