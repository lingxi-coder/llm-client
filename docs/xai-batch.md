# xAI 原生 Batch API

`XaiBatchService` 封装 xAI 文档中的批次容器、批量添加请求、状态读取、分页列举、取消和分页结果读取。xAI REST 流程分为创建批次和添加请求两个操作，因此 `create()` 与 `submit()` 是分开的。每个方法只执行一次 HTTP 请求；服务不会自动重试、轮询或继续抓取后续页。

```rust,no_run
use lingxi_llm_client::{
    protocol::Secret,
    transport::Transport,
    xai_batch::{
        XaiBatchChatCompletion, XaiBatchCreateRequest, XaiBatchInput,
        XaiBatchPageOptions, XaiBatchRequest, XaiBatchRequestBody,
        XaiBatchScope, XaiBatchService,
    },
};
use serde_json::json;

async fn xai_batch_example(
    transport: &dyn Transport,
    api_key: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let scope = XaiBatchScope::new(
        "xai-production",
        "account-42",
        "https://api.x.ai/v1",
    )?;
    let batch = XaiBatchService::new(transport, Secret::new(api_key), scope)?;
    let created = batch
        .create(&XaiBatchCreateRequest::new("nightly-evaluation")?)
        .await?;
    let requests = XaiBatchInput::new(vec![XaiBatchRequest::new(
        "case-001",
        XaiBatchRequestBody::ChatGetCompletion(XaiBatchChatCompletion::new(
            "grok-4.3",
            vec![json!({"role":"user", "content":"Summarize this report."})],
        )?),
    )?])?;
    batch.submit(&created.reference, &requests).await?;

    let _status = batch.get(&created.reference).await?;
    let page = batch
        .results(&created.reference, &XaiBatchPageOptions::new().limit(100))
        .await?;
    let _result_count = page.results.len();
    Ok(())
}
```

`XaiBatchScope` 固定 profile、账户和完整 API endpoint。标准 endpoint 是 `https://api.x.ai/v1`；文档列出的美国区域 endpoint 是 `https://us.api.x.ai/v1`。从一个 scope 得到的 batch 引用不能用于另一个 profile、账户或 endpoint。

文件模式下，调用 `FileService::upload(..., FilePurpose::Batch)` 上传 JSONL，再通过 `XaiBatchInputFileRef::from_uploaded_file(&scope, &file)` 绑定返回的 `ProviderFileRef`，并将其传给 `XaiBatchCreateRequest::with_input_file`。Files adapter 接受使用普通 OpenAI Chat 或 Responses protocol 的 xAI profile；Batch 请求本身仍使用原生、与模型无关的 xAI 路由。上传按官方示例只发送 multipart `file` 字段，不发送可选的 OpenAI 兼容 `purpose`。类型化引用保留精确的 profile、endpoint、账户 scope 以及文件过期和就绪状态元数据；scope 不匹配或文件已过期时，会在创建 Batch 前拒绝。文件模式的批次创建后即封存，`submit()` 会在本地拒绝。

当前 xAI Files REST 上传参考规定最大 50 MB，而 Batch 指南另行宣传文件批次最大 200 MB。客户端保守采用 Files endpoint 的 50,000,000 字节限制，不声称已验证 200 MB 上传。xAI 还说明单个文件最多包含 50,000 个请求；本客户端不会下载或检查调用方拥有的 JSONL，因此行数和行内容仍由服务端校验。上传文件的保留和清理由调用方负责。

示例使用官方标明支持 Batch 的 `grok-4.3`。[Grok 4.7](https://docs.x.ai/developers/models/grok-4.7)、4.6、4.5 与 Grok Build 0.1 的模型卡明确标明 Batch 不可用，这些已知模型及已确认的别名会在构造请求时拒绝；未知模型仍由提供方校验。

批量请求通过 `batch_request_id` 对应输入和结果；xAI 要求同一批次里的 ID 唯一。服务会拒绝空 ID 和同一次添加中的重复 ID；若对同一批次分多次调用 `submit()`，调用方还需保证新 ID 不与先前提交的重复。当前类型覆盖官方文档展示的 `chat_get_completion`、`responses`、`image_generation`、`image_edit`、`video_generation` 和 `video_extension` 请求变体。Chat、Responses 和媒体请求的扩展参数使用 `with_parameter` 保留。`batch_request_id` 的实际边界由 xAI 服务端决定；客户端不擅自要求特定字符集。

`submit()` 调用 `POST /v1/batches/{batch_id}/requests`，其成功响应按文档可以为空。输入中单个请求遵守 xAI 文档规定的 25 MiB 载荷上限；为了限制一次序列化的内存占用，此服务还将单次 inline 添加请求数量限制为 100,000 条、整体 JSON 限制为 256 MiB。后两项是本客户端边界；很大的作业可由宿主分成多次 `submit()`。

批次状态包括 `num_requests`、`num_pending`、`num_success`、`num_error` 和 `num_cancelled`。xAI 文档将 `num_pending == 0` 说明为所有已添加请求均已处理；快照保留完整 provider 原始 JSON。`list()`、`list_requests()` 和 `results()` 每次只返回一页。`XaiBatchPageOptions` 的 `limit` 接受 1–1000，可将响应中的 `pagination_token` 传给下一次调用。

结果页保留完整原始响应。可识别的成功响应、错误消息、取消和待处理状态有独立 outcome；未来或当前未识别的结果形状会保留为 `Other`，不会被误报成成功。结果按 `batch_request_id` 匹配。xAI 图片和视频结果中的签名 URL 文档称约一小时后过期，宿主应及时保存需要保留的媒体。

`create()`、`submit()` 和 `cancel()` 发出后若传输或响应读取中断，服务返回 `OutcomeUnknown`，不会重复提交。`submit` 与 `cancel` 的未知错误会附上已知的 scope 引用，调用方可以用 `get()`、`list_requests()` 或 `results()` 对账。确定的非 2xx 响应仍作为 provider 错误返回。

参考：[xAI Batch API 指南](https://docs.x.ai/developers/advanced-api-usage/batch-api)、[Batch REST API 参考](https://docs.x.ai/developers/rest-api-reference/inference/batches)和[Files 上传参考](https://docs.x.ai/developers/rest-api-reference/files/upload)。
