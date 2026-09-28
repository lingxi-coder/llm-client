# Qwen 百炼文本重排

[English](qwen-rerank.en.md)

`QwenRerankService` 为阿里云 Model Studio 的 provider-native `qwen3.7-text-rerank` HTTP API 提供同步封装。调用方提交一个 query 和候选文档列表，得到按相关性排序的原始文档索引、分数和可选 token 用量。该接口与兼容 OpenAI 的 `qwen3-rerank` `/compatible-api/v1/reranks` 路由是不同契约；本模块只实现原生接口。

当前官方 [Text Rerank API](https://help.aliyun.com/en/model-studio/text-rerank-api) 明确列出的原生 HTTP 路由是：

```text
POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
```

请求使用 Bearer Model Studio API Key，正文采用 `model`、`input.query`、`input.documents` 和可选 `parameters.top_n` / `parameters.instruct`。成功结果位于 `output.results`；每项的 `index` 指向原始输入 `documents`，`relevance_score` 的范围是 0 到 1。服务保留 Model Studio 返回的顺序和原始索引，不会改写候选文本。

调用方必须明确传入 profile、稳定且非敏感的账户标识、地域和 workspace ID，并在 `RequestOptions` 中提供同一账户标识和该地域的 API Key。workspace ID 用作请求主机名的一部分，因此不接受多个 DNS 标签或路径。该原生 HTTP 参考目前只给出了北京的精确路由；虽有通用地域 URL 说明，但没有为此模型和路由列出其他地域的完整地址，本实现不会推测新加坡或其他地域的地址。

```rust,ignore
use lingxi_llm_client::{
    providers::qwen::rerank::{QwenRerankRegion, QwenRerankRequest, QwenRerankScope, QwenRerankService},
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

服务会在发送前检查 query、候选文档、最多 500 个文档、`top_n` 和账户作用域。Model Studio 文档还列出了每个 query/document 最多 30,000 tokens，并建议单次输入不超过 120,000 tokens；本 crate 不估算服务端 tokenizer，超限由服务端拒绝。`top_n` 大于文档数时，Model Studio 会返回全部文档。

每次调用只发送一个 HTTP 请求，不会自动重试。传输中断或服务端 5xx 会以 `Unknown` 表示处理结果不确定；HTTP 4xx 表示请求被拒绝；服务端已返回 2xx 但响应结构无效时标记为 `Accepted`，调用方不应将其解释成安全重放的依据。契约测试仅使用 mock transport，不会访问或计费真实账户。
