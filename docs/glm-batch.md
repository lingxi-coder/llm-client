# GLM Batch 批处理

[English](glm-batch.en.md)

`client.provider::<ZhipuClient>(profile)?.batch(scope)` 提供智谱 GLM 原生 Batch 的受限生命周期：上传 JSONL 输入、提交 24 小时任务、单次读取状态、列出一页任务、请求取消，以及流式读取成功或错误结果文件。当前只开放智谱官方 Java SDK 示例中的 `/v4/chat/completions` endpoint；JSONL 行使用 `custom_id`、`method`、`url`、`body` 字段。官方示例使用 `24h` completion window。参考[智谱官方 SDK 的 Batch 示例](https://github.com/MetaGLM/zhipuai-sdk-java-v4#batch-processing)、[智谱 API 介绍](https://docs.bigmodel.cn/cn/api/introduction)、[列出批处理任务 API](https://docs.bigmodel.cn/api-reference/批处理-api/列出批处理任务)和[官方 OpenAPI 规范](https://docs.bigmodel.cn/openapi/openapi.json)。

服务固定使用中国大陆 BigModel API 根 `https://open.bigmodel.cn/api/paas/v4`。`GlmBatchRegion::International` 会在创建 scope 时被拒绝：当前[官方 Z.AI 文档索引](https://docs.z.ai/llms.txt)没有发布 Batch lifecycle，因此本模块不把中国区路由或凭据用于国际区。智谱官方 SDK 文档列出的同步和 Batch 示例支持此接口族，但该模块不把 Z.AI 国际端点视作已验证的 Batch 服务。

```rust,no_run
use lingxi_llm_client::{
    providers::zhipu::batch::{
        GlmBatchChatRequest, GlmBatchInput, GlmBatchLine, GlmBatchMessage,
        GlmBatchMetadata, GlmBatchRegion, GlmBatchScope,
    },
    protocol::Secret,
    LlmClient,
};
use serde_json::json;

async fn submit_glm_batch(
    client: &LlmClient,
    api_key: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let scope = GlmBatchScope::new(
        "glm-mainland",
        "zhipu-account-42",
        GlmBatchRegion::ChinaMainland,
    )?;
    let provider = client.provider::<lingxi_llm_client::providers::zhipu::ZhipuClient>(scope.profile_name())?;
    let request_options = lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..Default::default()
    };
    let service = provider.batch(scope)?;
    let body = GlmBatchChatRequest::new(
        "glm-5.3",
        vec![GlmBatchMessage {
            role: "user".into(),
            content: json!("Summarize this record."),
        }],
    )?;
    let input = GlmBatchInput::new(vec![GlmBatchLine::new("record-1", body)?])?;
    let input_file = service.upload_input(&input, &request_options).await?;
    let job = service
        .submit(&input_file, &GlmBatchMetadata::default(), &request_options)
        .await?;
    Ok(job.reference.batch_id().to_owned())
}
```

输入行必须使用唯一 `custom_id`，且一个文件只能包含同一模型的请求。消息 `content` 和其它模型参数保留 JSON 形状；请求不能启用 `stream`。客户端在本地将输入限制为最多 50,000 行、单行 1 MB、总计 200 MB。

上传通过 multipart `purpose=batch` 发送。成功上传会产生 `GlmBatchInputFileRef`，只有同一 profile、账户、区域和 endpoint scope 下的服务才能提交它。提交会创建 provider job；`GlmBatchJobRef` 和 output/error file references 带同样的 scope 指纹，不能跨账户或 endpoint 复用。调用 `get` 是一次状态读取，`list` 读取一页任务并支持官方 `after` 游标和 `limit` 参数；如 `has_more` 为真，可把 `last_id` 传给下一次显式调用。`cancel` 是一次取消请求，`stream_result` 只打开指定 JSONL 文件流。轮询、后续分页和 JSONL 业务解析由宿主决定。

上传、提交和取消都不会自动重试。如果连接中断、provider 返回 5xx，或 provider 返回成功但响应无法解析，结果会标成 `OutcomeUnknown` 或 `OutcomeUnknownResponse`。提交可能已被接收，调用方应先通过控制台或 provider 查询核对后再决定是否重提。明确的 4xx 错误作为 provider rejection 返回。下载结果直接产出原始字节块，不在内存中聚合整个文件。测试使用 mock transport，不会调用真实或付费服务。
