# Anthropic Messages Batch

`AnthropicBatchService` 将 Anthropic Messages Batch API 封装为独立服务。它以内联 `requests` 数组提交请求、读取和列举批次状态、请求取消、删除已完成的批次，并在结果响应到达时逐条解析记录。服务不会写入 JSONL 文件，也不会把完整结果集缓存在内存中。

通过 `AnthropicClient::batch(scope)` 创建的绑定服务，在每次操作中使用 profile 注册的鉴权器和调用时提供的凭证；直接构造的独立服务继续使用 Anthropic API key。

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

// `transport` implements the crate's Transport trait. API key 保存在宿主的
// 凭证存储中，并在每次操作时通过 RequestOptions 传入。
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

每个 `custom_id` 在批次内必须唯一，并匹配 `[a-zA-Z0-9_-]{1,64}`。类型化核心字段要求模型、至少一条消息和 `max_tokens > 0`；可通过 `with_parameter` 添加 `system`、`tools`、`thinking` 等 Messages 参数。Anthropic Batch 不支持 `stream: true` 和 `speed`，服务会在发送前拒绝这些参数。服务也会检查当前文档规定的上限：每批 100,000 条请求，创建请求体不超过 256 MiB。

`create`、`get`、`list`、`cancel` 和 `delete` 每次只发送一个请求。Anthropic 只允许删除已完成的批次；若批次仍在处理，应先取消，再通过 `get` 确认终态。轮询时机和失败项的后续处理由调用方决定。创建、取消或删除请求发出后若传输或读取响应失败，服务会返回“结果未知”错误，不会自动重试；成功 HTTP 响应的正文无法解析时也会标为未知。明确的非 2xx 响应则作为 provider 错误返回。

`AnthropicBatchRef` 绑定 provider、profile、endpoint 指纹和账户／workspace 范围。其他范围的服务会在发送网络请求前拒绝此引用。Anthropic 返回的 `results_url` 会保留在快照供查看，但结果请求仍由引用绑定的 API endpoint 和批次 ID 构造，避免跟随响应中任意提供的 URL。

`stream_results` 每次产出一个 `AnthropicBatchItemResult`。结果顺序可能与输入不同，应使用 `custom_id` 对应原请求。成功消息、单项错误、取消、过期和未知的新结果类型都有独立类型。单项错误仍作为流中的一条结果，不会挡住后续记录。传输错误、格式错误的行或超过 16 MiB 的单行会终止结果流并返回错误。

调用方负责安排轮询、保存结果以及决定是否重新提交失败项。Anthropic 当前文档说明批次最长可处理 24 小时，结果从创建起可读取 29 天。本服务不会自动轮询、自动重试，也不会在创建服务时调用在线 provider。

参考：[Anthropic Batch processing 指南](https://platform.claude.com/docs/en/build-with-claude/batch-processing)、[Create a Message Batch](https://platform.claude.com/docs/en/api/messages/batches/create)和[Delete a Message Batch](https://platform.claude.com/docs/en/api/http/messages/batches/delete)。
