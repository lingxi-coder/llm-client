# Qwen 批量推理

`qwen_batch` 提供阿里云百炼 Model Studio Batch API 的类型化接口。它使用官方文档列出的 OpenAI 兼容路由：北京 `https://dashscope.aliyuncs.com/compatible-mode/v1`，新加坡 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。两个地域的 API Key 不可互换。

当前接口覆盖文本 Chat Completions 和 Embeddings 输入，并提供 JSONL 上传、任务提交、查询、单页任务列表、取消，以及成功结果和错误文件的流式下载。输入行由类型化请求构成，不接受自定义 HTTP 方法或 URL；一个输入文件必须使用相同 endpoint、模型和（适用时）`enable_thinking`。当前类型不表示多模态内容、工具调用或 Qwen 的测试 endpoint。

```rust,no_run
// `client` and `api_key` come from the host application.
# async fn example(client: &lingxi_llm_client::LlmClient, api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    protocol::Secret,
    providers::qwen::batch::{
        QwenBatchChatMessage, QwenBatchChatRequest, QwenBatchChatRole, QwenBatchCompletionWindow,
        QwenBatchInput, QwenBatchLine, QwenBatchListOptions, QwenBatchMetadata, QwenBatchRegion,
        QwenBatchRequestBody, QwenBatchScope, QwenBatchSubmitOptions,
    },
};

let scope = QwenBatchScope::new(
    "qwen-production",
    "account-123",
    QwenBatchRegion::Beijing,
    Some("workspace-456".into()),
)?;
let provider = client.provider::<lingxi_llm_client::providers::qwen::QwenClient>(scope.profile_name())?;
let request_options = lingxi_llm_client::RequestOptions {
    credential: Some(Secret::new(api_key)),
    ..Default::default()
};
let service = provider.batch(scope)?;
let input = QwenBatchInput::new(vec![QwenBatchLine {
    custom_id: "case-1".into(),
    body: QwenBatchRequestBody::Chat(QwenBatchChatRequest {
        model: "qwen-plus".into(),
        messages: vec![QwenBatchChatMessage {
            role: QwenBatchChatRole::User,
            content: "用一句话介绍杭州。".into(),
        }],
        temperature: None,
        top_p: None,
        max_tokens: Some(128),
        enable_thinking: Some(false),
    }),
}])?;
let file = service.upload_input(&input, &request_options).await?;
let options = QwenBatchSubmitOptions {
    completion_window: QwenBatchCompletionWindow::from_hours(24)?,
    metadata: QwenBatchMetadata {
        name: Some("nightly-eval".into()),
        description: None,
    },
};
let job = service.submit(&file, &options, &request_options).await?;
let _page = service.list(&QwenBatchListOptions::new().limit(10), &request_options).await?;
let latest = service.query(&job.reference, &request_options).await?;
if let Some(output) = latest.output_ref() {
    let mut bytes = service.stream_result(&output, &request_options).await?;
    // 消费者按需读取 bytes 流中的 JSONL 内容。
}
if let Some(errors) = latest.error_ref() {
    let mut bytes = service.stream_result(&errors, &request_options).await?;
    // 消费者按需读取 JSONL 错误详情。
}
# Ok(())
# }
```

调用方负责等待任务进入终态；服务不会自动轮询、重试或保存文件。Batch 输入最多 50,000 行、500 MB，每行最多 1 MB，`custom_id` 必须唯一且不超过 256 个字符。完成窗口限于 24 到 336 小时。官方 API 只允许查询最近 30 天内创建的任务。查询响应必须标识请求中的同一任务；不同的任务 ID 会被拒绝。取消成功可能先返回 `cancelling`；之后可继续查询。若成功的取消响应返回了不同任务 ID，服务会报告取消结果未知，且不会重试。快照中的 `file-batch_output-` 和 `file-batch_error-` ID 分别可通过 `output_ref()` 与 `error_ref()` 获取引用，并由 `stream_result()` 访问官方 `/files/{file_id}/content` 路由。

`list` 每次只读取一页，并支持官方 `after`、`limit`、任务名称、输入文件 ID、状态和创建时间过滤。`limit` 范围为 1 到 100，时间过滤使用 `yyyyMMddHHmmss`。若 `has_more` 为真，可将 `last_id` 作为下一次显式请求的 `after`。服务会在返回前拒绝无效过滤条件、重复任务 ID 和未推进的分页游标。

文件、任务和结果引用都绑定 provider、profile、HTTP endpoint fingerprint、account scope、地域和可选 workspace ID。错误账户、连接、地域或 workspace 的引用会在发送网络请求前被拒绝。workspace ID 目前只参与本地引用隔离：官方 Batch 文档未说明 workspace 专属 URL 或 header，因此本模块不会自行拼接或发送它。调用方仍须为声明的地域和 workspace 提供相应 API Key。

输入范围、文件限制、Batch 生命周期以及结果文件路由见阿里云官方[批量推理指南](https://help.aliyun.com/zh/model-studio/batch-inference)和[OpenAI 兼容 Batch API](https://help.aliyun.com/zh/model-studio/batch-interfaces-compatible-with-openai)。
