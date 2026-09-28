# xAI Speech-to-Speech Realtime

`realtime::XaiRealtimeSession` 将 xAI 原生 `/v1/realtime` WebSocket 协议接入现有的 `RealtimeTransport` 和有界实时队列。宿主通过连接请求提供认证，并在自己的事件循环中同时运行返回的 `RealtimeDriver`。默认音频传输是 JSON/Base64；输入和输出也可分别配置为原始二进制 WebSocket 帧。

启用 `realtime-websocket` feature 后可使用内置 `HttpTransport`。直接会话 endpoint 为 `wss://api.x.ai/v1/realtime?model=grok-voice-latest`；xAI 也文档化了 `grok-voice-think-fast-2.0`。认证通过 `Authorization: Bearer ...` 请求头传入。对于客户端直连，xAI 文档说明可使用短期 client secret，放在同一 Bearer 请求头或 `Sec-WebSocket-Protocol: xai-client-secret.<token>` 中。请勿将 API key 或 client secret 写入日志。

```rust,ignore
use std::sync::Arc;
use lingxi_llm_client::realtime::{
    RealtimeConnectRequest, RealtimeInput, RealtimeLimits, RealtimeTransport,
    HttpTransport, XaiRealtimeConfig, XaiRealtimeSession,
};

let transport: Arc<dyn RealtimeTransport> = Arc::new(HttpTransport::new()?);
let request = RealtimeConnectRequest {
    endpoint: "wss://api.x.ai/v1/realtime?model=grok-voice-latest".into(),
    headers: vec![("Authorization".into(), format!("Bearer {xai_token}"))],
    max_frame_bytes: 1024 * 1024,
};
let (session, driver) = XaiRealtimeSession::connect(
    transport,
    request,
    XaiRealtimeConfig::default(),
    RealtimeLimits::default(),
).await?;
let (control, mut events) = session.into_parts();
// 在宿主的 executor 上并发运行 `driver.run()`。
control.send(RealtimeInput::Text("Hello".into()))?;
while let Some(event) = events.next().await {
    // 处理有类型的 xAI 会话、文本、音频、转录和响应事件。
}
```

连接后，服务端会发送 `session.created` 和 `conversation.created`。适配器收到 `session.created` 后发送唯一的一条 `session.update`，再等待 `session.updated` 才返回会话。初始原生事件会按顺序重放到返回的事件流中。`XaiRealtimeEvent` 为常见会话生命周期、语音活动、转录、音频/文本输出、响应、函数参数、MCP 和错误事件提供类型化结构。未识别或新增事件会以 `XaiRealtimeEvent::Native` 保留原始 JSON；核心关闭与连接事件则包装在 `XaiRealtimeEvent::Realtime` 中。

文本输入会先发送包含 `input_text` 片段的 `conversation.item.create`，再发送 `response.create`。默认音频传输通过 Base64 JSON `input_audio_buffer.append` 发送，默认格式为小端序 PCM16、24 kHz；适配器也支持 xAI 文档列出的 PCM 采样率、G.711 μ-law/A-law 和 24 kHz 单声道 Opus 数据包。输入和输出格式、JSON 或二进制传输都可独立配置。二进制输入以 `RealtimeFrame::Binary` 发送；二进制输出通过 `XaiRealtimeEvent::AudioBinaryDelta` 返回。该二进制事件没有 JSON 响应/项目标识；JSON 音频 delta 则继续通过 `AudioDelta` 保留原生事件和索引。

`XaiRealtimeConfig` 还可设置文档化的 reasoning effort（`high`/`none`）、server VAD threshold（0.1–0.9）、静音和前置填充时长（各不超过 10,000 ms）、idle timeout、输出语速（0.7–1.5）、发音替换表以及 client-side function 工具声明。工具 JSON Schema 由调用方提供；工具授权和执行仍由宿主负责。VAD 选项仅能与 server VAD 一起使用，客户端会在建立连接前拒绝越界配置。

设置 `input_transcription = true` 可请求 `grok-transcribe` 输入转录，也可附加 BCP-47 `input_transcription_language_hint` 和最多 100 项、每项最多 50 个字符的 `input_transcription_keyterms`。提示和关键词都必须与转录同时启用。xAI 发送的 `conversation.item.input_audio_transcription.updated` 是累计文本，后续更新可能修订之前的结果，随后会发送最终完成事件。

默认的 `XaiTurnDetection::ServerVad` 由 xAI 检测语音结束，因此音频只发送 append 事件，调用 `CommitAudio` 会被拒绝。设置 `XaiTurnDetection::Manual` 会把 `turn_detection.type` 设为 `null`；此时 `CommitAudio` 会发送 `input_audio_buffer.commit`，`Interrupt` 会发送 `response.cancel`。`ClearAudio` 或 `XaiRealtimeCommand::ClearAudioBuffer` 会发送 `input_audio_buffer.clear`，丢弃尚未提交的输入音频。两种模式下，文本输入都会显式请求生成响应。

`XaiRealtimeControl::send_command` 还支持 xAI 的 `ForceMessage` 扩展（直接合成固定文本，不需要 `response.create`）和带单次 instructions 覆盖的 `CreateResponse`。这些 provider 命令和通用 `RealtimeInput` 共用同一个有界发送队列和帧上限。xAI 的文档列出删除与截断会话项事件，但没有在 Realtime 页面给出对应请求字段；本适配器会对这些 generic 输入返回 `InvalidInput`，不会猜测并发送字段。`conversation.item.retrieve` 则被 xAI 明确列为不支持。

每个 `XaiRealtimeEvent` 都通过 `native` 字段保留服务端原始事件 JSON。例如，`AudioDelta` 提供解码后的字节、响应/项目标识、内容索引和已配置的输出格式；`InputTranscriptionUpdated` 提供 xAI 的累计转录文本，后续更新可能修订之前的内容。音频设备访问、播放、工具授权和会话策略仍由宿主负责。

### 返回自定义函数结果

自定义函数由宿主执行。收到 `FunctionCallArgumentsDone` 后，将服务端给出的 `call_id` 和执行结果发回。`ToolResult` 提交单条结果；`ToolResults` 按输入顺序批量提交，并作为一个有界队列项发送。两种方式都会把结果序列化为文档规定的 `output` JSON 文本，但都不会自动请求下一轮。只有当本轮每个函数调用都有结果，且宿主准备好让 xAI 继续发言时，再发送 `ContinueResponse`：

```rust,ignore
use lingxi_llm_client::realtime::{RealtimeInput, RealtimeToolResult};

control.send(RealtimeInput::ToolResults {
    results: vec![
        RealtimeToolResult { call_id: first_call_id, output: first_result },
        RealtimeToolResult { call_id: second_call_id, output: second_result },
    ],
})?;

// 宿主可以等当前响应的音频播放完成后再继续。
control.send(RealtimeInput::ContinueResponse)?;
```

这遵循 xAI 并行工具调用规则：先提交所有 `function_call_output`，再发送唯一一次 `response.create`。每批需包含 1–128 个不重复、非空的 call ID；128 是本客户端的队列保护上限，不是 xAI 限额。配置的 frame 大小也限制结果编码大小。无效批次会在进入队列前失败。如果发送过程中传输中断，客户端不会自动重放；宿主应将送达状态视为不确定。

启用 `resumption_enabled` 时必须提供稳定且非 secret 的 `credential_scope`。宿主可保存 `ConversationCreated` 事件中的 `conversation.id`，然后用 `XaiRealtimeResumeRef::import(id, model, endpoint, credential_scope)` 创建作用域引用，并在重连配置中传入 `resume_ref`。连接前会检查引用的模型、规范化 endpoint route 和账户作用域，并将 `model`/`conversation_id` 写入 WebSocket URL。引用的 `Debug` 不输出 conversation ID 或账户范围。服务端会重放历史，但客户端不会自动重连或重发任何未确认输入；历史在 30 分钟无活动后过期。宿主管理连接生命周期，并决定如何处理原生函数调用事件。回归使用 fake transport，不会调用真实 xAI 账号或产生付费请求。

参考：[xAI Voice WebSocket API 参考](https://docs.x.ai/developers/rest-api-reference/inference/voice)、[Speech-to-Speech 指南](https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech) 和 [xAI Voice 概览](https://docs.x.ai/developers/model-capabilities/audio/voice)。
