# MiniMax 双向流式 TTS

`minimax_bidi_tts` 实现 MiniMax 原生 `t2a_v2_bidi` 双向 WebSocket 生命周期。国际版官方路由为 `wss://api.minimax.io/ws/v1/t2a_v2_bidi`，大陆版为 `wss://api.minimax.cn/ws/v1/t2a_v2_bidi`。服务使用调用方注入的 `RealtimeTransport`，每次连接单独传入凭证，并根据显式区域选择路由。

```rust,no_run
# async fn example(
#     transport: std::sync::Arc<dyn lingxi_llm_client::realtime::RealtimeTransport>,
#     api_key: String,
# ) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    minimax_bidi_tts::{
        MiniMaxBidiTtsConfig, MiniMaxBidiTtsCredentials, MiniMaxBidiTtsError,
        MiniMaxBidiTtsEventKind, MiniMaxBidiTtsLanguageBoost, MiniMaxBidiTtsLimits,
        MiniMaxBidiTtsParameters, MiniMaxBidiTtsRegion, MiniMaxBidiTtsRequest,
        MiniMaxBidiTtsService,
    },
    protocol::Secret,
};

let service = MiniMaxBidiTtsService::new(
    transport,
    MiniMaxBidiTtsConfig::new(
        "voice-profile",
        "team/account-17",
        MiniMaxBidiTtsRegion::ChinaMainland,
    ),
)?;
let credentials = MiniMaxBidiTtsCredentials::new(Secret::new(api_key));
let mut parameters = MiniMaxBidiTtsParameters::new("English_expressive_narrator");
parameters.language_boost = Some(MiniMaxBidiTtsLanguageBoost::Chinese);
let request = MiniMaxBidiTtsRequest::new("English_expressive_narrator")
    .with_parameters(parameters)
    .with_session_id("host-session-17");

// connect 建立会话并发送 task_start。
let session = service
    .connect(&credentials, &request, MiniMaxBidiTtsLimits::default())
    .await?;
let (mut input, mut events) = session.into_parts();
input.continue_text("Hello from a streaming text source.").await?;
// 由宿主安排保活；注入的传输可能不支持显式 Ping。
match input.ping(Vec::<u8>::new().into()).await {
    Ok(()) | Err(MiniMaxBidiTtsError::PingUnsupported) => {}
    Err(error) => return Err(error.into()),
}
input.finish().await?;
let mut task_finished_seen = false;
while let Some(event) = events.next().await? {
    match event.kind {
        MiniMaxBidiTtsEventKind::AudioDelta { audio, is_final, .. } => {
            let _audio_chunk = audio;
            let _request_audio_is_final = is_final;
        }
        MiniMaxBidiTtsEventKind::TaskFinished => task_finished_seen = true,
        MiniMaxBidiTtsEventKind::TaskFailed { .. } => {
            return Err(std::io::Error::other("MiniMax task failed").into());
        }
        MiniMaxBidiTtsEventKind::SoftError { .. } => {
            // 是否重新发送由宿主决定；客户端不会自动重放文本。
        }
        _ => {}
    }
}
if !task_finished_seen {
    return Err(std::io::Error::other("session ended without task_finished").into());
}
# Ok(())
# }
```

`MiniMaxBidiTtsParameters` 复用普通 MiniMax WebSocket T2A 的类型化任务设置；Bidi 请求另外支持可选的客户端 `session_id`。官方 schema 列出八种模型，以及嵌套的 voice/audio 设置、发音词典、混合音色、音色修改、字幕、语言增强和 `continuous_sound`。通用设置校验复用普通 TTS 类型；普通 WSS 路由的限制不会套用到 Bidi。

建立连接后，客户端等待 `connected_success`，发送一次 `task_start`，再等待 `task_started` 才返回会话。连接截止时间涵盖握手和这两个就绪事件。发送 `task_start` 之前发生的错误属于连接或协议错误；写入已开始但没有收到确认时，结果会标为未知。客户端不会自动重试或重连。

每次收到文本片段时都可调用 `task_continue`。客户端不会裁剪、规范化、补标点或拆分文本；空字符串、空白、换行和标点都会原样保留。MiniMax 会静默丢弃仅含空白的文本，并把任意粒度的输入缓冲成句子。每个片段最多可有 10,000 个 Unicode 字符；默认本地上限与官方上限相同，发送前还会检查序列化后的帧大小。

事件流区分三个完成层次：音频的 `is_final`、句子的 `sentence_end`、以及任务的 `task_finished`。音频以十六进制形式放在 `data.audio` 中，客户端会将每段解码为字节，并保留对应原生 JSON、`extra_info`、`session_id`、`trace_id` 和 `connect_id`。句子边界作为独立事件返回。未知 JSON 事件会连同完整原生值返回；解码失败的事件保留原值，无效 JSON 和二进制帧则在 `raw_frame` 中保留原始字节。

`input.flush()` 请求 MiniMax 合成已缓冲的尾部文本，但保持会话打开。发送更多文本前，应先读到 `TaskFlushed`。`input.cancel()` 会打断当前合成，包括尚未完成的 flush；继续发送前应先读到 `TaskCanceled`。取消等待期间到达的迟到 `TaskFlushed` 仍会作为事件返回，但不会解除取消门控。两种控制操作都不会重试或重发内容。

输入端与事件端可以并行推进。如果控制确认在对应的发送操作返回前到达，客户端会先保留该确认；只有发送本身成功后，才会开放输入门控。若发送失败，客户端会报告结果未知，不会声称操作已成功。

如果 `continue_text`、`flush`、`cancel` 或 `finish` 的 future 在帧写入期间被取消，客户端会将会话设为终态，因为无法确认 MiniMax 是否已收到该帧。之后的输入会返回 `Closed`；客户端不会自动重试，也不会把会话恢复为可写状态。可显式调用 `abort()` 清理会话。若事件读取正等待服务端帧，终态通知会唤醒读取并释放读写两端。

`input.finish()` 请求 MiniMax 合成剩余文本并结束任务。仅仅成功发送请求不代表任务完成。若要将 EOF 视为业务成功，必须先收到有效的 `TaskFinished`，再遇到干净的 WebSocket EOF。`TaskFailed` 后的终止 EOF 也可能让 `events.next()` 返回 `None`，但这代表任务失败，不能视为成功；示例会在收到 `TaskFailed` 时返回错误。`TaskFinished` 之前的 EOF，以及该事件之后发生的传输错误，都会作为错误报告。`abort()` 只在本地关闭连接，不表示任务完成。

服务商错误码 2204 和 2205 会以 `SoftError` 事件返回，不会关闭会话。是否重新发送文本由宿主决定；客户端不会重放。其他非零服务商状态码和 `task_failed` 事件属于终止失败；客户端会主动关闭 WebSocket，返回一次 `TaskFailed`，后续事件读取以 `None` 结束，调用方必须保留失败结果。官方指南说明，2204 表示超出字符限制的文本片段被跳过；2205 表示服务端待合成队列容量已满。

连接空闲约 120 秒后，MiniMax 可能主动关闭连接。宿主可自行安排调用 `input.ping(payload)`，发送真实的 WebSocket Ping 控制帧，负载最多 125 字节；它不会被序列化成 JSON。客户端不提供自动 Ping 定时器或重连。注入的传输必须支持显式 Ping；默认实现会返回 `PingUnsupported`，保活时机由应用负责。

`connect_with_voice` 只接受与当前服务的 provider、profile、account scope、region 和官方 HTTP `/v1` 根地址都匹配的 `MiniMaxVoiceRef`。客户端会在建立 WSS 连接前检查引用，并将其 voice ID 写入 `task_start`。account scope 是调用方提供的元数据，并不能证明 API key 实际属于该账号。

`MiniMaxBidiTtsLimits` 限制收发帧大小、文本字符数、单会话返回的累计音频字节数，以及连接就绪阶段可接收的未识别事件数。这些是客户端的内存和安全上限；只有 10,000 字符文本限制也来自 MiniMax 官方文档。客户端不会解码音频 codec、组装 Opus 容器、播放音频，也不会决定保活间隔。

官方资料：[国际版 Bidi T2A WebSocket](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket-bidi)、[大陆版 Bidi T2A WebSocket](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket-bidi)、[MiniMax 音色生命周期](minimax-voices.md)和[普通 MiniMax WebSocket TTS](minimax-streaming-tts.md)。
