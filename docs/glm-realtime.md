# GLM Realtime（语音 WebSocket）

本模块实现智谱 GLM Realtime 官方 WebSocket 协议的服务端接入路径。它只连接官方文档确认的中国大陆端点 `wss://open.bigmodel.cn/api/paas/v4/realtime`；没有根据普通 HTTP API 推导国际站实时端点。

依据：[MetaGLM 官方 Realtime SDK 协议说明](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/GLM-Realtime-doc-for-llm.md)。文档确认 JSON/WebSocket、Bearer header、`session.update`、WAV Base64 音频输入、PCM/MP3 音频输出，以及 `response.audio.*`、`response.audio_transcript.*` 和 `response.done` 事件。

## 连接与账户范围

每次连接都显式提供区域路由、非密钥账户范围和当前凭证。凭证只作为本次 WebSocket 握手的 `Authorization: Bearer …` header 传入；API Key 与宿主签发的 JWT 均由宿主管理。`GlmRealtimeScope` 保存 profile 名、账户身份、区域和 endpoint fingerprint，不保存凭证。

```rust,no_run
use std::sync::Arc;
use lingxi_llm_client::{protocol::Secret, realtime::{
    GlmRealtimeRoute, GlmRealtimeScope, GlmRealtimeSession, GlmRealtimeConfig,
    RealtimeLimits, RealtimeTransport,
}};
async fn connect(
    transport: Arc<dyn RealtimeTransport>,
    credential: Secret<String>,
) -> Result<(), Box<dyn std::error::Error>> {
let route = GlmRealtimeRoute::mainland_china();
let scope = GlmRealtimeScope::new("glm-mainland", "billing-account-1", &route)?;
let (session, driver) = GlmRealtimeSession::connect(
    transport,
    route,
    scope,
    credential, // 每次连接传入的 Secret<String>
    GlmRealtimeConfig::default(),
    RealtimeLimits::default(),
).await?;
Ok(())
}
```

`connect` 等待服务器发送 `session.created`，随后发送官方 `session.update` 配置，并等待 `session.updated` 后才返回。会话创建期间收到的事件会留在接收队列中。客户端可用同一 executor 并行运行返回的 `RealtimeDriver` 和收发控制/事件。

默认会话使用 `input_audio_format: "wav"`、`output_audio_format: "pcm"`、`turn_detection.type: "server_vad"`，并显式设置 `beta_fields.chat_mode: "audio"`、`tts_source: "e2e"` 与 `auto_search: false`。可选 instructions 和 PCM/MP3 输出格式都有类型化配置。

## 输入与事件

- `RealtimeInput::Audio` 只接受 `RealtimeAudioFormat::Encoded { mime_type: "audio/wav" }`。数据可以是 WAV 字节块；驱动将其 Base64 编码到 `input_audio_buffer.append`。原始 PCM 输入不受此适配器支持。
- `server_vad` 下服务端负责检测、提交音频并触发音频轮次响应。此模式下 `CommitAudio` 会在本地拒绝。宿主仍可在提交函数结果后用 `ContinueResponse` 显式发送 `response.create`。
- `client_vad` 下，宿主发送音频后用 `CommitAudio` 发出 `input_audio_buffer.commit`，再用 `ContinueResponse` 发出 `response.create`。
- `RealtimeInput::Text` 按 GLM 协议发送 `conversation.item.create`（`input_text`），随后发送 `response.create`。
- 使用 `GlmRealtimeConfig.tools` 配置由宿主执行的函数工具。GLM session 采用扁平字段结构（`type`、`name`、可选 `description` 与 JSON Schema `parameters`）。`tool_choice` 支持 `auto`、`none`、`required`，或 `{ "type": "function", "function": "tool_name" }`。
- `response.function_call_arguments.done` 映射为 `FunctionCallArgumentsDone`，并保留原始 native event。由于公开协议说明的事件表和示例都没有 `call_id`，该字段保持可选。只有事件提供有效 `call_id` 时才提交结果；适配器不会生成关联 ID。`ToolResult` 和 `ToolResults` 会发送 `conversation.item.create` 的 function 输出项，并原样使用调用者提供的 `call_id` 和紧凑 JSON 输出。它们不会自动启动响应；宿主处理完函数调用后应单独发送 `ContinueResponse`。
- `Interrupt` 映射到 `response.cancel`。设备采集、VAD 判定、播放与音频排队属于宿主职责。
- 输出音频的 `response.audio.delta` 解码为字节并保留 response/item/index 字段。PCM/MP3 类型取自本次会话设置。官方协议说明没有为 PCM 指定采样率，本模块不会猜测或填充采样率。
- GLM 文档定义的文本输出是 `response.audio_transcript.delta` 与 `response.audio_transcript.done`，映射为 `TextDelta`/`TextDone`。官方说明该转写由独立模型产生，可能与生成内容有差异或为空；它不是权威文本。`response.done` 保留完整 native response，其中包括 provider 给出的 usage。
- `error` 作为带 native JSON 的 `ProviderError` 交给宿主；官方称多数事件错误不会关闭会话。未知事件保留为 `Native`。

GLM 当前公开的 Realtime 文档不确认国际站 WebSocket endpoint；本模块不提供国际路由。这里需要保留官方资料中的差异：协议说明的事件表和示例没有 `call_id`，而当前 [MetaGLM Python SDK model](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/python/rtclient/models.py) 在函数参数完成事件与函数输出项中都定义了可选 `call_id`。协议说明的[函数输出示例](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/GLM-Realtime-doc-for-llm.md)同样没有该 ID，[SDK 示例 handler](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/python/samples/message_handler.py)也没有演示它。本适配器遵循 SDK schema，同时保留事件 native JSON；本地协议测试不代表服务端已实测接受。二进制 WebSocket frame 与自动搜索工具执行仍不在适配器范围内。服务端断线会报告 `ConnectionInterrupted` 并结束 driver；不会自动重连或重放输入。

`RealtimeInput::ClearAudio` 发送 `input_audio_buffer.clear`，丢弃尚未提交的输入音频。确认事件保留为原生提供方事件。此操作不取消输出或关闭会话；`FinishSession` 仍不支持。

契约来源：[client events](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/GLM-Realtime-doc-for-llm.md).
