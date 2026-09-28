# xAI Deferred Chat

[English](deferred.en.md)

先用 `client.provider::<XaiClient>(profile)?` 绑定具体 profile，再通过 `provider.deferred()` 调用资源。每次操作传入 `RequestOptions`，client 不保存凭证。

`provider.deferred()` 封装 xAI [Deferred Chat Completions](https://docs.x.ai/developers/advanced-api-usage/deferred-chat-completions) 的单次提交和单次结果读取。内置 `grok` Chat profile 配有独立的 Deferred 路由；`grok-responses` 不使用这一 Chat 专有接口。配置 v3 可以继承或显式禁用路由。服务不会轮询、重试或切换连接。

提交时使用 `ChatRequest`，并在 `RequestOptions.account_scope` 提供同一账户稳定、非密钥的标识。客户端复用所选 Chat codec 编码，加入 `deferred: true`，返回绑定 provider、profile、提交端点、结果端点、账户和模型的 `DeferredJobRef`。当前不支持 Responses 续传或托管工具组合；应用附件若尚未解析会在发送前拒绝。普通函数工具的后续执行仍由宿主负责。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::ChatRequest;
use lingxi_llm_client::providers::xai::deferred::{DeferredError, DeferredJob, DeferredJobRef, DeferredPoll};

async fn submit(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<DeferredJob, Box<dyn std::error::Error>>
{
    let provider = client.provider::<lingxi_llm_client::providers::xai::XaiClient>("grok")?;
    provider.deferred().submit(request, options).await.map_err(Into::into)
}

async fn check(client: &LlmClient, ticket: DeferredJobRef, options: &RequestOptions)
    -> Result<DeferredPoll, Box<dyn std::error::Error>>
{
    let provider = client.provider::<lingxi_llm_client::providers::xai::XaiClient>(&ticket.profile_name)?;
    provider.deferred().fetch_once(ticket, options).await.map_err(Into::into)
}
```

`fetch_once()` **消耗**句柄：HTTP `202` 返回 `DeferredPoll::Pending(ticket)`，供宿主稍后再次查询；HTTP `200` 返回 `Completed`，包含解析后的 `ChatResponse` 和完整原生 JSON，且不再返回可重用句柄。模型目录在排队期间发生变化时，读取仍使用句柄固定的原始模型 ID。xAI 规定完成结果在 24 小时内只能读取一次，因此宿主应在收到 `Completed` 后立即持久化需要的数据。提交传输中断返回 `SubmitOutcomeUnknown`；提交已接受但缺少有效 ID 返回 `SubmitAcceptedUnknownId`，两者都不能盲目重提。读取中断返回含原句柄的 `FetchOutcomeUnknown`，但服务端可能已经消耗结果，客户端不会自动重查。已消耗或过期时保留提供方的 HTTP 错误状态和响应体，不伪装成待处理状态；解析失败另行返回。

测试使用 mock；未执行真实账户验收。[xAI REST 参考](https://docs.x.ai/developers/rest-api-reference/inference/chat-completions)。
