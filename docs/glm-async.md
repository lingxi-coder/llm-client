# GLM 异步对话补全

`glm_async` 为智谱 BigModel 提供单次异步对话提交和结果查询。大陆官方文档列出了 `POST /api/paas/v4/async/chat/completions` 与 `GET /api/paas/v4/async-result/{id}`；提交参数沿用类型化 `ChatRequest` 和 OpenAI Chat codec。调用方负责安排查询间隔、持久化引用及处理已完成结果。

```rust,ignore
use lingxi_llm_client::{
    glm_async::{GlmAsyncConfig, GlmAsyncCredentials, GlmAsyncRegion},
    protocol::Secret,
};

let snapshot = client.snapshot();
let profile = snapshot.provider("glm-mainland").expect("configured GLM profile").clone();
let config = GlmAsyncConfig::new(
    profile,
    GlmAsyncRegion::ChinaMainland,
    "tenant-7/zhipu-account",
    "zhipu-key-slot-2",
)?;
let service = snapshot.glm_async(config)?;
let credentials = GlmAsyncCredentials::new(
    "zhipu-key-slot-2",
    Secret::new(api_key),
);

let accepted = service.submit(&chat_request, &credentials).await?;
// Persist accepted.reference. The host decides when to query again.
let latest = service.get(&accepted.reference, &credentials).await?;
println!("status: {:?}; provider payload: {}", latest.task_status, latest.native);
```

`GlmAsyncConfig` 显式绑定 profile、区域、账户和非机密凭证槽标识；每次操作仍单独传入 API key。任务引用会绑定 provider、profile、区域、端点、账户、凭证槽与模型，跨作用域查询会在发请求前拒绝。凭证槽必须与 `GlmAsyncCredentials` 中的标识相同。序列化引用不含 API key。

提交返回带原生 JSON 的任务快照。`PROCESSING`、`SUCCESS` 和 `FAIL` 映射为已知状态，其余状态原样保留为 `Other`；响应中的 `error` 另存于 `native_error`。查询只发一次 GET，不会轮询、重提或切换账户。提交期间发生传输中断、HTTP 408/5xx 或成功响应缺少任务 ID 时，返回 `SubmitOutcomeUnknown` 并保留已有 HTTP 状态、请求 ID 与原生响应；调用方应先自行确认任务状态，不能盲目重提。

异步 Chat 文档没有声明取消操作，因此此服务不提供 `cancel`。释放请求 future 也不代表服务端任务已取消。

大陆 API 参考：[异步对话补全](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8%E5%BC%82%E6%AD%A5)、[查询异步结果](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%9F%A5%E8%AF%A2%E5%BC%82%E6%AD%A5%E7%BB%93%E6%9E%9C)、[官方文档索引](https://docs.bigmodel.cn/llms.txt)。

国际 profile 可以通过 `GlmAsyncRegion::International` 明确选择，但当前 Z.AI 官方文档只列出同步 Chat Completion，未列出异步 Chat 路由；因此配置会在发送前以 `UnsupportedRegion` 拒绝，不猜测国际端点。智谱 API 的国内外账户和密钥也不可互换。官方补充异步路由文档后再开放该区域。
