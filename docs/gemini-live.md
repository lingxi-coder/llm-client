# Gemini Live WebSocket 会话

`realtime::GeminiLiveSession` 使用现有的 `RealtimeTransport` 建立 Google Gemini Live `BidiGenerateContent` 会话。宿主通过 endpoint 或请求头提供认证，运行返回的 `RealtimeDriver`，并负责工具授权、音频设备和播放状态。

内置 `RustlsWebSocketTransport` 需要启用 `realtime-websocket` feature。宿主也可以注入自己的 `RealtimeTransport`。

Google 原始 WebSocket endpoint 为 `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`。标准 API key 可放在 `?key=...` 查询参数中。Google 文档还说明，短期 token 可通过 `access_token` 查询参数或 `Authorization: Token ...` 请求头传入。请勿将这些值写入日志。

```rust,ignore
use std::sync::Arc;
use lingxi_llm_client::realtime::{
    GeminiLiveConfig, GeminiLiveEvent, GeminiLiveSession, RealtimeConnectRequest,
    RealtimeInput, RealtimeLimits, RealtimeTransport, RustlsWebSocketTransport,
};

let transport: Arc<dyn RealtimeTransport> = Arc::new(RustlsWebSocketTransport);
let mut config = GeminiLiveConfig::new("gemini-3.8-live", "google-account-a");
config.system_instruction = Some("Answer briefly.".into());
config.enable_session_resumption = true;

let (session, driver) = GeminiLiveSession::connect(
    transport,
    RealtimeConnectRequest {
        endpoint: format!("{google_live_websocket_url}?key={api_key}"),
        headers: vec![],
        max_frame_bytes: 1024 * 1024,
    },
    config,
    RealtimeLimits::default(),
).await?;
let (control, mut events) = session.into_parts();
// Run `driver.run()` concurrently on the host executor.
control.send(RealtimeInput::Text("Hello".into()))?;
while let Some(event) = events.next().await {
    match event {
        GeminiLiveEvent::SessionResumptionUpdate(update) => {
            // Persist update.handle when it is Some.
        }
        GeminiLiveEvent::Realtime(event) => {
            // Consume normalized audio/text/tool events under host policy.
        }
    }
}
```

`connect` 先发送唯一的一条 `setup` JSON 消息，并等待 Google 返回 `setupComplete` 后才返回会话。该消息设置模型、生成参数、可选的纯文本系统指令、工具和实时输入配置。此 endpoint 不会收到 OpenAI Realtime 格式的事件。

文本会编码为 `realtimeInput.text`。音频输入要求宿主提供单声道、小端序 PCM16、16 kHz 数据；适配器检查声明的 PCM 格式和采样率，然后用 `audio/pcm;rate=16000` MIME 类型以 Base64 发送。请将麦克风数据切成较小的数据块。Google 默认启用自动活动检测，此时 `CommitAudio` 会发送 `audioStreamEnd: true`。若设置 `realtimeInputConfig.automaticActivityDetection.disabled = true`，文本输入会用 `activityStart` 和 `activityEnd` 括起，`Interrupt` 会发送 `activityStart`，而 `CommitAudio` 会发送 `activityEnd`。自动检测启用时调用 `Interrupt` 会返回 `InvalidInput`，因为 API 只允许在关闭自动检测后由客户端发送活动信号。

输出音频和文本会映射为 `AudioDelta` 与 `TextDelta`。`turnComplete` 映射为 `TurnCompleted`，`interrupted` 映射为 `Interrupted`。函数调用映射为 `ToolCall` 事件。发送 `ToolResult` 时，必须使用该事件中的调用 ID，并提供 JSON 对象；codec 会保留对应的函数名，以构造 Google 的 `toolResponse.functionResponses` 消息。工具授权和执行仍由宿主负责。

设置 `enable_session_resumption = true` 可接收恢复状态更新。服务端的 `sessionResumptionUpdate` 会映射为类型化的 `GeminiLiveResumeUpdate`；只有当 Google 将状态标记为可恢复并提供非空 `newHandle` 时，更新才包含句柄。句柄可序列化，供宿主管理持久化；其 `Debug` 输出会隐藏服务商 token。句柄绑定模型、endpoint 的 origin/path，以及宿主提供的非机密 `credential_scope`。endpoint 查询参数不会写入绑定信息，因此不会在那里保存 API key 或短期 token。请为每个 Google 账号使用稳定的 credential scope，并只在模型、endpoint 和账号 scope 都一致时复用句柄。scope 不匹配时会在建立网络连接前拒绝操作；如果查询参数还承载路由语义，请用 credential scope 区分对应配置。

Google 文档说明，恢复句柄在会话终止后两小时内有效；生成过程或函数调用期间等部分状态无法恢复。客户端不会自动重试、重连、重放输入或恢复会话。宿主决定是否以及何时携带最新的可恢复句柄开启新会话。其他服务端消息（包括 `goAway` 和工具调用取消）会以原生 `ProviderEvent` 数据保留。

启用恢复会让 Google 侧保存会话状态。Google 的零数据保留说明称，生成了会话句柄时，关联对话状态最多可能保留 24 小时。对不允许保留的对话，请勿启用恢复。

测试使用本地 fake transport，没有调用 Google 在线服务。

参考：[Gemini Live WebSocket 快速入门](https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket)、[WebSockets API 参考](https://ai.google.dev/api/live)、[会话管理](https://ai.google.dev/gemini-api/docs/live-api/session-management)、[零数据保留说明](https://ai.google.dev/gemini-api/docs/zdr) 和 [Live 转录音频格式](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)。

`RealtimeInput::Image` 将非空 JPEG/PNG 编码为 `realtimeInput.video`；采集、编码与发送节奏由宿主控制，输入及编码后的帧沿用会话大小限制。

`RealtimeInput::ToolResults` 将结果作为一个 `toolResponse.functionResponses` 消息发送。所有结果必须对应已记录的调用 ID、使用 JSON 对象，且批次内 ID 唯一；取消通知移除对应调用。整批校验完成后才发送。后续生成由 Gemini 决定，`ContinueResponse` 仍不支持。新增功能等待统一测试，未调用真实 Google 接口。

函数结果通过帧大小校验并进入发送队列后，释放对应的调用名称记录。队列满、队列关闭或帧校验失败时保留记录，允许调用方稍后重试。入队不等于服务端已收到；传输失败仍终止驱动且不自动重发。
