# Qwen-Omni Realtime（WebSocket）

本模块实现阿里云百炼 Qwen-Omni Realtime 的 WebSocket 会话协议。当前路由覆盖官方接入指南列出的华北 2（北京）和新加坡两个地域，以及 Qwen3.8 Omni Flash、Qwen3.5 Omni Plus、Qwen3.5 Omni Flash 三个实时模型。

依据：[百炼 Realtime 接入文档](https://help.aliyun.com/zh/model-studio/realtime)、[客户端事件参考](https://help.aliyun.com/zh/model-studio/client-events)、[服务端事件参考](https://help.aliyun.com/zh/model-studio/server-events)。

## 区域、workspace 与凭证

`QwenRealtimeRoute::new` 要求一个地域和一个 workspace ID，并只构造对应的完整 WSS 路由：

- 北京：`wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=...`
- 新加坡：`wss://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=...`

连接时，当前调用的 API Key 通过 `Authorization: Bearer …` 握手 header 传入。API Key 必须与选定地域和 workspace 匹配，由宿主管理和逐次提供。`QwenRealtimeScope` 将 profile、账户、region、workspace 和 endpoint fingerprint 绑定；凭证不会进入 scope。

```rust,ignore
let route = QwenRealtimeRoute::new(QwenRealtimeRegion::Beijing, workspace_id)?;
let scope = QwenRealtimeScope::new("qwen-mainland", "billing-account-1", &route)?;
let config = QwenRealtimeConfig::new(QwenRealtimeModel::Qwen38OmniFlashRealtime);
let (session, driver) = QwenRealtimeSession::connect(
    transport,
    route,
    scope,
    api_key, // 每次连接传入的 Secret<String>
    config,
    RealtimeLimits::default(),
).await?;
```

连接后，驱动等待 `session.created`，发送 `session.update` 并等到 `session.updated` 后才返回。模型名同时放入官方要求的 URL `model` 查询参数和会话配置。配置可选择文本输出或文本加音频输出；官方不支持 audio-only。模型默认输入是 16 kHz 单声道 PCM16，输出是 24 kHz PCM16。`semantic_vad` 适用于这组已列出的实时模型；手动模式将 `turn_detection` 设为 `null`。

## 输入与输出

- 音频输入使用 `RealtimeInput::Audio`，格式必须为 `Pcm16 { sample_rate_hz: 16000 }`。它编码为 Base64 `input_audio_buffer.append`。
- 图片/视频帧使用 `RealtimeInput::Image`，只接受 `image/jpeg`，并限制每帧 Base64 大小不超过 256 KiB。Qwen 通过 `input_image_buffer.append` 接收逐张 JPEG 帧；每帧前必须至少成功排入一个音频 append。百炼建议约每秒一帧。采集、抽帧、节奏控制属于宿主。
- 文本输入使用 `conversation.item.create` 添加 `input_text` 用户消息，再显式发送 `response.create`。
- Server VAD 自动提交音频并生成响应；手动模式要求宿主在音频后调用 `CommitAudio`，再调用 `ContinueResponse`。VAD 模式下这两个显式操作会在本地拒绝，避免重复提交或生成。`Interrupt` 映射为 Qwen 的 `response.cancel`。
- WebSocket 音频输出作为 PCM16、24 kHz 字节增量交给宿主；`response.audio_transcript.*` 与文本模式的 `response.text.*` 分别保留为独立事件。`response.done` 保留完整 native response 和 usage。
- `error` 作为 `ProviderError` 返回，不会自动结束会话；未知事件保留为 native JSON。底层远端断开会报告 `ConnectionInterrupted` 并停止驱动，不自动重连或重放输入。

本适配器不做设备采集/播放、视频抽帧、自动重连，也不封装 Function Calling/MCP 的工具执行和结果回传。图像与音频帧受 session frame 上限和发送队列约束；队列满时，调用方收到错误并决定如何处理。

`RealtimeInput::ClearAudio` 发送 `input_audio_buffer.clear`，丢弃尚未提交的输入音频。确认事件保留为原生提供方事件。此操作不取消输出或关闭会话；`FinishSession` 仍不支持。

契约来源：[client events](https://help.aliyun.com/en/model-studio/client-events).
