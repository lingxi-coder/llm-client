# OpenAI Responses 后台任务

[English](background.en.md)

`client.background()` 对接 OpenAI [Background mode](https://developers.openai.com/api/docs/guides/background)。内置 `openai` profile 独立配置 Responses 后台路由；配置 v3 可继承或禁用。提交沿用选定模型的 Responses 编码和请求校验，加入 `background: true`。接口提供非流式提交、单次查询与取消，以及可恢复的原生事件流和 provider-neutral 解码事件流；轮询时间、句柄持久化及工具执行由宿主负责。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::ChatRequest;
use lingxi_llm_client::background::{BackgroundError, BackgroundJob, BackgroundJobRef};

async fn submit(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<BackgroundJob, BackgroundError>
{
    client.background().submit("openai", request, options).await
}

async fn check(client: &LlmClient, reference: &BackgroundJobRef, options: &RequestOptions)
    -> Result<BackgroundJob, BackgroundError>
{
    client.background().get(reference, options).await
}
```

流式提交使用 `background=true, stream=true`。`submit_stream().next_event()` 每次返回原生 JSON 和游标；宿主处理事件后保存游标，断线后传给 `resume_stream()`。恢复请求使用 `stream=true&starting_after=<sequence_number>`，不会重提任务。首个包含 response ID 的事件到达前没有可用游标；此时断线意味着提交结果未知。流式错误保留最后交付的游标，收到终态事件后流结束。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::ChatRequest;
use lingxi_llm_client::background::{BackgroundError, BackgroundEventCursor, BackgroundStreamError};

async fn stream_job(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<Option<BackgroundEventCursor>, BackgroundError>
{
    let mut stream = client.background().submit_stream("openai", request, options).await?;
    let mut last = None;
    loop {
        match stream.next_event().await {
            Ok(Some(event)) => last = Some(event.cursor),
            Ok(None) => break,
            Err(BackgroundStreamError::Interrupted { cursor, .. }) => {
                // Persist cursor; reconnect later with client.background().resume_stream(cursor, options).
                last = cursor.map(|cursor| *cursor);
                break;
            }
            Err(_) => break,
        }
    }
    Ok(last)
}
```

需要 provider-neutral 输出时，使用 `submit_chat_stream()` 和 `resume_chat_stream()`。每个事件会保留原生 JSON 和游标，并附上解码后的 `StreamEvent`。终态 `response.completed` 或 `response.incomplete` 事件还会提供重建后的 `ChatResponse`。`response.failed` 仍作为错误返回，并保留原生事件和游标。此前交付的增量已经由调用方收到，属于调用方持有的部分结果。恢复后的解码器从游标之后开始；若需要连续的事件历史，请由调用方保留恢复前已收到的增量。终态事件包含完整 response，因此恢复后仍可独立重建完整 `ChatResponse`。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::{ChatRequest, ChatResponse};

async fn read_response(
    client: &LlmClient,
    request: &ChatRequest,
    options: &RequestOptions,
) -> Result<Option<ChatResponse>, Box<dyn std::error::Error>> {
    let mut stream = client.background().submit_chat_stream("openai", request, options).await?;
    while let Some(event) = stream.next_event().await? {
        // 处理事件后持久化 event.cursor。若宿主需要厂商事件字段，可保留 event.native。
        if let Some(response) = event.response {
            return Ok(Some(response));
        }
    }
    Ok(None)
}
```

`RequestOptions.account_scope` 必须提供同一账户稳定、非机密的标识。返回的 `BackgroundJobRef` 绑定 provider、profile、端点、账户、模型和 response ID；查询与取消在发送前校验作用域。`queued` 和 `in_progress` 仍在运行；`completed`、`failed`、`cancelled`、`incomplete` 是终态。成功及不完整响应在可解码时提供 `ChatResponse`；完成的响应还可得到同账户续传引用。所有状态都保留原生 JSON。失败状态保留原生 `error`，不伪装成成功文本。

提交网络中断产生 `SubmitOutcomeUnknown`，不可盲目重提；取消中断产生带句柄的 `CancelOutcomeUnknown`，应查询状态核对。客户端不自动重试、轮询或切换连接。OpenAI 文档说明，未显式持久保存的后台响应通常只在约 10 分钟的轮询窗口内保留；宿主应及时获取并保存所需结果。尚未使用真实账户验收。参见 [后台模式与流恢复](https://developers.openai.com/api/docs/guides/background)、[Responses 查询](https://developers.openai.com/api/reference/typescript/resources/responses/methods/retrieve) 和 [取消](https://developers.openai.com/api/reference/go/resources/beta/subresources/responses/methods/cancel)。

## 删除已保存的 Response

`background().delete(&reference, &options)` 单次发送 `DELETE /responses/{id}`，只有返回的 `id`、`object: "response"` 和 `deleted: true` 全部匹配才返回 `BackgroundDeletion`。回执保留作用域引用和原生字段。删除与取消推理是独立操作，由宿主决定何时删除持久化结果。它沿用查询的 provider/profile/endpoint/account 校验，不要求原模型仍在当前目录中。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions, background::BackgroundJobRef};
async fn remove_saved_response(
    client: &LlmClient,
    reference: &BackgroundJobRef,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let deleted = client.background().delete(reference, options).await?;
    assert_eq!(deleted.reference.response_id, reference.response_id);
    Ok(())
}
```

传输或读取失败、服务端错误、无效的成功回执均返回带引用和原始错误的 `DeleteOutcomeUnknown`。客户端不重试，也不把 `404` 推断为本次删除成功；由调用方核对状态后决定后续操作。[官方删除接口](https://developers.openai.com/api/reference/cli/resources/responses/methods/delete)。
