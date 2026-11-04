# 会话增量续接与恢复

`ConversationSession` 是可选的会话 helper。每次仍向 SDK 提供完整、可重放的 `ChatRequest`；helper 在签名和发送前检查历史前缀，将实际请求缩减为上一响应引用与新增输入。它不持久化历史，也不执行工具。

当前接入：OpenAI Responses、xAI Responses HTTP/WebSocket、Qwen Responses、Gemini Interactions、第一方 Anthropic Message threads，以及独立的智谱国内 `glm-responses` 路由。DeepSeek、MiniMax、OpenRouter 与未证实的兼容网关保持全量请求。服务端存储和提示缓存分别配置。

## 使用

每个独立会话持有一个 helper。fork、撤销或压缩历史时调用 `reset()`，或创建新 helper。`account_scope` 必须由宿主提供，并稳定标识真实账户；没有账户作用域时不做 HTTP 隐式续接。

```rust,no_run
use lingxi_llm_client::{ConversationSession, LlmClient, RequestOptions};
use lingxi_llm_client::protocol::{ChatRequest, ConversationMessage, LlmError};

async fn example(client: &LlmClient, options: &RequestOptions) -> Result<(), LlmError> {
    let mut session = ConversationSession::new();
    let mut request = ChatRequest::new("YOUR_CONFIGURED_MODEL");
    request.messages.push(ConversationMessage::user_text("记住编号 4921。"));

    let first = session.complete_in(
        &client.snapshot(), "YOUR_PROFILE", &request, options,
        || Ok(()), // 每次物理发送前的同步准入/预算检查
        |response| { let _usage = response.usage_report(); }, // 每个响应先记账
    ).await?;

    request.messages.push(first.message);
    request.messages.push(ConversationMessage::user_text("编号是多少？"));
    let _second = session.complete_in(
        &client.snapshot(), "YOUR_PROFILE", &request, options,
        || Ok(()), |_| {},
    ).await?;
    let _diagnostic = session.last_request(); // 模式、logical/wire 字节数、新增项数
    Ok(())
}
```

`stream_in()` 返回原有 `ModelStream`；继续使用 `next()` 或 `next_batch()`、`StreamAccumulator` 和现有用量接口。只有有效终止事件能提交续接状态。提前丢弃流、取消、解析失败或不完整输出，不会把半轮响应当作下一轮的历史边界。

宿主若已管理 prepared-call 生命周期，可使用 `session.prepare(&mut draft)`、`draft.seal()`、`session.dispatch(call, on_dispatch)`。这是单次发送接口；收到错误后先保留记账数据，再用 `session.recover(collected.response())` 判断是否允许以完整历史重新准备。每个恢复请求都要重新签名和准入，不能复用旧 `PreparedCall`。

## 恢复边界

高层 helper 最多恢复一次，且留在相同 snapshot、端点和账户。完整重放不重新执行本地工具。明确的 continuation/thread 失效、线程竞争或指纹错误允许重建；定向指出当前引用 ID 不存在的响应也可恢复。普通模型 404、限流、网络超时、断流或已开始生成后的错误不自动重提。

流式接口可恢复 HTTP 错误响应，或首个非 ping SSE/WS 事件中的明确续接拒绝；被探测的字节会交回原来的响应观察路径。返回流之后不再隐式重放生成。一次完整恢复失败会把错误交给调用方，避免循环重试。完成请求默认采用同一 120 秒总预算；显式 `total_timeout` 覆盖该预算，并跨恢复共享。

`last_request()` 只报告尺寸和模式，不暴露正文、凭证或原始响应 ID。helper 不推断 token 免费、缓存命中或实际加速幅度。

## Provider 差异

- **Anthropic**：仅在已声明能力的第一方端点使用 `message-threads-2026-08-12`。create 发送完整输入；continue 省略 system/tools。自动缓存、工具结果、固定字段变化、refusal/fallback 和流式工具 JSON 均有独立处理。线程空闲五分钟后重建。`thread_unsupported_request` 会在本轮最多降级一次，并为相同身份停用 thread；不删除原请求的其他设置。当前 context-management/compaction 请求使用全量模式。工具与 MCP 配置仍经过现有完整请求校验。
- **OpenAI / Qwen**：显式 `store=false` 不产生持久续接引用；服务端返回的存储关闭也会被尊重。instructions 保留在每轮请求中。Qwen 的 session-cache header 是额外可选优化。
- **xAI**：预设声明 Responses WS 能力；有 WebSocket transport 时复用连接，`store=false` 只使用连接内状态。无 WS transport 或握手要求 HTTP 时可在生成前回退 HTTP。重连后缺少持久引用时发送完整历史。无法无损进入现有 WS JSON 表示的输入使用 HTTP。
- **Gemini**：统一 codec 的输入是 `input.steps`。每轮重发 tools、system_instruction 和 generation_config。终止帧省略 steps 时，使用 decoder 已验证、完整装配的原始步骤；不会从显示文本猜测历史。`ConversationSession` 仅为 `provider_id = "google"` 且 base URL 为 `https://generativelanguage.googleapis.com/v1beta` 的官方端点启用跨模型续接。同一连接与账户内切换模型后，完整历史前缀必须一致，且目标模型的 `metadata.inputModalities` 必须明确支持历史内容的全部模态（包括旧输出和工具结果）。当前识别 text、image、audio、video 和 PDF；能力未知、其他原生步骤或内容类型会保守地发送完整历史。除模型名外的请求设置变化仍触发全量请求。参见官方[跨模型兼容要求](https://aistudio.google.com/docs/interactions-overview#limitations)。
- **OpenRouter**：预设不启用远端续接，不论是否切换模型都发送完整历史。其官方 [Responses API 文档](https://openrouter.ai/docs/api_reference/responses/overview) 明确声明无状态；`store=true` 或非空 `previous_response_id` 会返回 400。API schema 出现同名参数并不代表支持存储或引用历史，上游模型厂商的能力也不能直接继承。
- **智谱国内**：`glm-responses` 是隐藏、显式选择的国内计量路由，使用平台 API Key，初始模型为 `glm-5.3`。必须设置 `request.controls.responses.store = Some(true)` 才启用续接；不会替用户开启存储。初始适配范围是文本、多模态基础输入、普通 function 工具及续接；可调 effort 与 hosted tools 没有从 OpenAI 路由直接继承。国际 Z.AI/Coding Plan 不受此开关影响。

首轮输出和后续历史必须保留 reasoning、signature、provider-native items、工具调用/结果及 phase 等原始语义。helper 遇到无法确认的前缀会发送全量，而不会猜测裁剪边界。

## 切换模型与缓存

切换模型不删除宿主持有的完整历史。Anthropic thread 绑定模型，OpenAI/xAI/Qwen/GLM 的跨模型引用兼容性未获明确保证，因此仍从完整历史开始新的续接起点。Gemini 按上述兼容检查决定是否续接。引用被明确拒绝时，仍最多以新模型和完整历史恢复一次；账户准入、签名与记账照常执行。

这里的跨模型优化仅适用于提供完整可重放历史的 `ConversationSession`。显式 `ChatRequest.continuation` / `ContinuationRef` 没有完整历史可供验证，仍要求模型一致，不会放宽其账户、端点或 workspace 绑定。服务端 prompt/KV cache 是否命中不由 interaction/response ID 保证；本地消息投影缓存也应按协议与实际内容判断能否复用，不应仅因模型名改变就清空。

官方来源与核查矩阵见[实施准备文档](conversation-continuation-plan-2026-11-02.md)。合同测试位于 `tests/conversation_session.rs`，另复用 `continuation`、`responses_session`、`gemini_interactions_codec`、`interactions`、`presets` 等回归。离线 mock 通过不代表真实账户、生产发布或性能验收。

最终组合回归共 954 项通过，详见[验证记录](conversation-continuation-validation-2026-11-03.md)。原有普通 chat 调用保持原行为；宿主需要显式持有并调用 `ConversationSession`，本次未更新 Harness 的固定依赖修订。
