# OpenRouter Batch 批处理

[English](openrouter-batch.en.md)

`OpenRouterBatchService` 实现 OpenRouter 的 inline Batch API：调用 `POST /api/v1/batches`，在一个 JSON 数组中提交多条请求。它不上传文件、不生成 JSONL，也不单独下载结果文件；查询批次时会直接返回结果。

类型化 API 覆盖 Chat Completions、Responses、Anthropic Messages 和 Embeddings。Chat、Responses 与 Messages body 可包含 URL-only 图片/文件部分；Embeddings 输入仍为文本。每个批次只使用一个模型和一个 endpoint，`custom_id` 必须唯一。可以设置 `provider.only` 限定路由提供方；其他同步请求使用的路由偏好不会被发送。请求 body 使用明确的类型，不接受任意 JSON 透传。

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

调用一次 `submit()` 并保存返回的 `OpenRouterBatchJobRef`。提交过程中若连接中断，或服务已收到成功状态但无法解析确认响应，会返回 `OpenRouterBatchError::OutcomeUnknown`；先用 `list()` 核对批次，再决定是否重新提交。`get()` 只查询一次，不会自动轮询。批次和每条结果都会保留原生字段、用量、批次级错误以及单条请求错误；批次成功完成时仍可能有部分请求失败。

`delete()` 调用 OpenRouter 的删除接口，清除保留的请求与结果数据；只有批次进入终态后才能删除。OpenRouter 没有公开 Batch 取消接口，因此本 API 不提供 `cancel()`。删除期间若连接中断，结果可能未知；重试前应检查批次列表。

`list()` 支持官方 `created_after` 和 `created_before` 过滤，可传 Unix 秒数或 ISO-8601 日期/日期时间。两者同时提供时，起始时间必须严格早于结束时间。服务保留调用方传入的时间文本，并在 query string 中进行 URL 编码。

引用中会绑定 `openrouter` provider ID、连接名称、API endpoint 指纹、账户作用域、批次 ID、请求 endpoint 和模型。调用方传入 API key 和稳定、非敏感的账户标识；服务不会保存凭证。请求使用 `https://openrouter.ai/api/v1`，提交和删除都不会自动重试。

类型化 Chat、Responses 和 Anthropic Messages body 可在相应 provider 支持时引用公开 HTTP(S) 图片或文件 URL。Batch 多模态输入只接受 URL：data URI、内联文件字节、provider 文件 ID、音频和视频都会被拒绝。单条请求是否可路由仍取决于具体 provider 和模型。单次序列化请求和单次响应均限制为 64 MiB，最多 50,000 条请求；若模型没有 Batch endpoint，服务端会拒绝提交。契约测试使用 mock，不代表已通过真实账户验收。

Chat Completions 使用 `OpenRouterBatchChatContent::Parts`，并通过 `OpenRouterBatchChatContentPart::image_url()` 或 `file_url()` 添加图片/文件。文件 helper 会把公开 URL 编码到官方 wire 字段 `file.file_data`。Responses 使用 endpoint 专属的 `OpenRouterBatchResponsesInput::Messages` 和 `OpenRouterBatchResponsesContentPart`。Anthropic Messages 使用 `OpenRouterBatchMessagesContent::Parts` 以及 `image_url()` 或 `document_url()`。URL 会按原样传给 provider；客户端不会下载文件，也不会把文件字节转换成 data URI。

若指定 `provider.only`，客户端会按 OpenRouter 当前能力表预检：图片 URL 支持列表包含 OpenAI、Anthropic 和 xAI，DeepInfra 仅支持 Chat Completions；文件 URL 支持列表包含 OpenAI（仅 Responses）、Anthropic、Mistral 和 DeepInfra（仅 Chat Completions），xAI 标记为不支持。`provider.only` 中的每个 provider 都必须支持请求的多模态内容。未指定 provider 时，路由由 OpenRouter 根据 provider 和模型能力处理。

`provider.only` 接受小写基础 provider slug 和官方的斜杠后缀 endpoint slug（例如 `deepinfra/turbo`），并将精确值保留在请求中。多模态预检只按已知基础 provider 判断能力；具体后缀和模型是否有可用 Batch endpoint 仍由 OpenRouter 判定。URL 预检只拒绝明显的本地/私有 IP literal 与本地域名，不进行 DNS 查询，也不证明公网 URL 可访问。

官方文档：[OpenRouter Batch API Quickstart](https://openrouter.ai/docs/batch-quickstart)、[Provider Selection](https://openrouter.ai/docs/guides/routing/provider-selection)、[OpenRouter Batch API 公告](https://openrouter.ai/blog/announcements/batch-api/)。
