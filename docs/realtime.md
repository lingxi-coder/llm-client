# Realtime 双向会话

`lingxi-llm-client::realtime` 提供与 provider 无关的低延迟双向会话接口。宿主负责运行驱动，并继续管理凭据、工具、音频采集与播放、权限和对话状态。默认不启用网络后端；启用 `realtime-websocket` feature 后可以使用基于 Rustls 的 `HttpTransport`。它要求 Tokio runtime 和 `wss://` endpoint，不会自动重连实时会话。

```rust,ignore
use std::sync::Arc;
use lingxi_llm_client::realtime::{
    OpenAiRealtimeCodec, RealtimeConnectRequest, RealtimeInput, RealtimeLimits,
    RealtimeSession, HttpTransport,
};

let transport = HttpTransport::new()?;
let (session, driver) = RealtimeSession::connect(
    &transport,
    RealtimeConnectRequest {
        endpoint: realtime_endpoint,
        headers: authenticated_headers,
        max_frame_bytes: 1024 * 1024,
    },
    Arc::new(OpenAiRealtimeCodec::default()),
    RealtimeLimits::default(),
).await?;
let (control, mut events) = session.into_parts();
// 在宿主执行器上并发运行 driver.run()。
control.send(RealtimeInput::Text("你好".into()))?;
while let Some(event) = events.next().await {
    // 处理标准化事件；ToolCall 仍由宿主按自身策略分发。
}
```

在依赖配置中启用 `features = ["realtime-websocket"]`，即可导入 `HttpTransport`。

内置传输使用 tokio-tungstenite、Rustls 和打包的 Mozilla 根证书完成 WebSocket 握手及 TLS，并按原样发送宿主提供的请求头。Tungstenite 处理 ping/pong；传输返回写入端和每次产出一个完整数据消息的输入流。传输不会在后台自动重试或重连，也不会从 endpoint 或请求头日志输出凭据。驱动被丢弃时会释放传输两端。未启用 `realtime-websocket` 时，宿主仍可注入自己的 `RealtimeTransport`。

出站队列有固定容量。`RealtimeControl::send` 和 `RealtimeControl::interrupt` 在队列已满时立即返回 `QueueFull`，麦克风循环可以丢弃或合并旧音频块，避免延迟不断累积。每个编码后的逻辑输入及每个入站帧都会受 `max_frame_bytes` 限制。事件队列也有界；宿主暂停读取事件时，驱动会暂停从传输层读取并施加反压。

调用 `close` 会把关闭命令排在之前已接受的消息之后，并等待传输层完成关闭。释放所有控制句柄后，驱动会发送正常关闭帧并发布 `Closed` 事件。直接丢弃驱动会通过释放注入的传输两端终止连接；Rust 的 `Drop` 无法保证异步完成 WebSocket 关闭握手。

远端主动关闭连接会作为连接中断报告；`Closed` 表示本地通过 `close` 或释放全部控制句柄完成的关闭。

连接前会生成并校验全部初始化帧。初始化配置错误或任一帧超过大小上限时，不调用传输层建立连接，也不发送部分初始化配置。

## OpenAI Realtime 编码器

`OpenAiRealtimeCodec` 在连接成功后先发送 `session.update`。模型和认证信息由宿主传入的 endpoint 与请求头决定。`OpenAiRealtimeConfig.tools` 可声明会话级函数工具，`tool_choice` 可选 `Auto`、`None`、`Required` 或指定已声明的 `Function(name)`。函数声明只暴露 Realtime 文档确认的 `type`、`name`、可选 `description` 和 JSON Schema `parameters`；名称重复、参数不是 JSON 对象或指定了未知函数时会在连接前拒绝。

文本输入会编码成用户会话条目和 `response.create`；音频块使用 `input_audio_buffer.append`；手动结束音频输入使用 `input_audio_buffer.commit`；`ClearAudio` 会发送 `input_audio_buffer.clear` 清除尚未提交的输入音频，服务端以 `input_audio_buffer.cleared` 确认。提交空缓冲区会由服务端报错，`commit` 本身不会创建模型响应；中断会发送 `response.cancel`。OpenAI Realtime WebSocket 没有 `FinishSession` 客户端事件，结束会话由宿主关闭 transport。图像输入编码成带 Base64 data URL 的 `input_image` 内容项，只使用宿主交给客户端的字节；客户端不会调用单独的文件上传 API 或读取远程 URL。图像只写入对话，宿主可在适当时机发送 `ContinueResponse`（`response.create`）来触发推理。OpenAI 当前文档确认 `gpt-realtime-2` 和 `gpt-realtime` 支持该图像输入，其他模型的可用性由服务端决定。`OpenAiRealtimeConfig.voice` 编码到 `session.audio.output.voice`，可选地指定具名内置语音（字符串）或项目可用的自定义语音 ID（`{ id }`）；名称不在客户端硬编码，是否可用由 OpenAI 项目和服务端校验。单个 `ToolResult` 保留便捷行为：发送一个 function-call output 条目后立即创建响应。若模型一次发出多个函数调用，宿主可用一个 `ToolResults` 输入一次入队全部结果；客户端只发送结果条目，随后由宿主在准备好继续时发送 `ContinueResponse`（`response.create`）。这让所有结果先进入对话，再开始下一轮推理。客户端不执行函数，也不决定何时继续。

默认会话使用单声道、小端序 24 kHz PCM16 输入和音频输出。OpenAI Realtime 参考文档列出了该采样率；传入的 [`RealtimeAudioFormat`] 必须与编码器配置相符。也可以配置 G.711 μ-law 或 A-law，以及纯文本输出。编码器会标准化音频与文本增量、已完成响应、函数调用、provider 错误和语音开始事件。尚未映射的 provider 事件会保留原始 JSON。图像编码后的数据仍受 `max_frame_bytes` 限制。

会话条目也支持显式 `RetrieveItem`、`DeleteItem` 与 `TruncateAudio` 操作，分别编码为 `conversation.item.retrieve`、`conversation.item.delete` 和 `conversation.item.truncate`。截断只适用于助手音频消息，OpenAI 要求 `content_index` 为 `0`；`audio_end_ms` 应由宿主按实际播放进度提供，超出原音频时长会由服务端报错。截断成功后服务端会同步删除未听到部分对应的文本。读取、删除和截断都不会自动创建模型响应。

工具调用只作为数据交给宿主授权和执行；本模块不会执行工具。`ToolResults` 每组至少一个、最多 128 个结果，并拒绝空或重复的 `call_id`；128 是客户端队列上限，不代表服务端限制。`response.cancel` 用于停止生成。如果宿主正在播放音频，还应停止本地播放并核对尚未听到的输出。OpenAI 文档说明 `output_audio_buffer.clear` 适用于 WebRTC/SIP，不适用于 WebSocket。

## Gemini Live 协议边界

Gemini Live 使用双向 WebSocket 和独立的 `BidiGenerateContent` 协议。`GeminiLiveSession` 会发送首条 `setup` JSON，并等待 `setupComplete` 后才把会话交给宿主。客户端消息使用 `realtimeInput` 或 `toolResponse`；服务端消息包括 `serverContent`、`toolCall`、中断和会话恢复事件。不要把 OpenAI 事件 JSON 发到 Gemini endpoint。详细接口见 [`gemini-live.md`](gemini-live.md)。

参考：[OpenAI Realtime 图像输入](https://developers.openai.com/api/docs/guides/realtime-conversations#image-inputs)、[OpenAI Realtime 自定义语音](https://developers.openai.com/api/docs/guides/custom-voices)、[OpenAI Realtime function tools](https://developers.openai.com/api/docs/guides/realtime-mcp#configure-a-function-tool)、[OpenAI Realtime 客户端事件](https://developers.openai.com/api/reference/resources/realtime/client-events)、[OpenAI Realtime 服务端事件](https://developers.openai.com/api/reference/resources/realtime/server-events)、[Gemini Live WebSocket API](https://ai.google.dev/api/live) 和 [Gemini Live WebSocket 快速入门](https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket)。

内置 WebSocket 后端不会自动恢复断线会话；此基础层也不包含真实账户验收、媒体采集与播放、视频帧支持或 provider 无关的工具执行。当前编码器只负责协议转换，不代表已验证真实 provider 服务可用。

实时语音转文字的独立 xAI `/v1/stt` 协议使用 [XaiSttSession](xai-stt.md)，支持原始二进制输入与按声道完成校验。

xAI `/v1/tts` 的双向文本转音频协议见[流式 TTS](xai-streaming-tts.md)，支持多轮合成、取消与发音替换表更新。
