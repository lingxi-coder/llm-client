# MiniMax WebSocket 流式 TTS

`minimax_streaming_tts` 实现 MiniMax 同步 Text-to-Audio WebSocket 协议。它允许调用方分段发送文本，并逐条接收十六进制编码的音频增量；这是文本转音频任务流，不是语音到语音的实时对话。

启用 `realtime-websocket` feature 可使用内置 Rustls WebSocket transport；宿主也可以注入自己的 `RealtimeTransport`。

```rust,ignore
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use std::sync::Arc;
use lingxi_llm_client::{
    minimax_streaming_tts::{
        MiniMaxStreamingTtsConfig, MiniMaxStreamingTtsEvent, MiniMaxStreamingTtsLimits,
        MiniMaxStreamingTtsRequest, MiniMaxStreamingTtsService,
    },
    minimax_tts::MiniMaxTtsRegion,
    protocol::Secret,
    realtime::{RealtimeTransport, RustlsWebSocketTransport},
};

let transport: Arc<dyn RealtimeTransport> = Arc::new(RustlsWebSocketTransport);
let service = MiniMaxStreamingTtsService::new(
    transport,
    MiniMaxStreamingTtsConfig::new(
        "minimax-voice",
        "team/account-17",
        MiniMaxTtsRegion::International,
    ),
)?;
let credential = Secret::new(api_key);
let session = service
    .connect(
        &credential,
        &MiniMaxStreamingTtsRequest::new("English_expressive_narrator"),
        MiniMaxStreamingTtsLimits::default(),
    )
    .await?;
let (mut input, mut events) = session.into_parts();

input.send_text("The first sentence.").await?;
while let Some(event) = events.next().await? {
    if let MiniMaxStreamingTtsEvent::AudioDelta { audio, is_final, .. } = event {
        // Consume or stream `audio` to the caller's playback or storage layer.
        let _ = audio;
        if is_final {
            break;
        }
    }
}
input.finish().await?;
while let Some(event) = events.next().await? {
    if matches!(event, MiniMaxStreamingTtsEvent::TaskFinished { .. }) {
        break;
    }
}
# Ok(())
# }
```

服务绑定到 profile、账号范围和明确地区。`International` 使用官方国际版 WSS 地址 `wss://api.minimax.io/ws/v1/t2a_v2`；`ChinaMainland` 使用大陆官方页面内嵌的 `wss://api.minimax.cn/ws/v1/t2a_v2`。构造器会按地区选择地址，服务也会拒绝与所选地区不完全匹配的 endpoint。每次连接都传入 `&Secret<String>`；服务不会保存凭据。

`minimax_voices` 返回的单个 `MiniMaxVoiceRef` 可传给 `connect_with_voice`。该方法会在建立 WSS 连接前检查 provider、profile、account scope、region 和 voice API endpoint：国际版只接受 `https://api.minimax.io/v1`，大陆版只接受 `https://api.minimax.cn/v1`，跨区引用会在网络连接前被拒绝。它不能与 `timbre_weights` 混用；混合音色应通过 `connect` 传入完整权重。account scope 是调用方提供的路由标签，不验证 API key 的实际归属；每次仍须传入正确账号的 key。内置声音 ID 可继续通过 `connect` 使用。

握手等待 `connected_success`，发送 `task_start`，再等待 `task_started`。两地区普通 T2A WebSocket schema 都列出 8 个模型：`speech-2.8-hd/turbo`、`speech-2.6-hd/turbo`、`speech-02-hd/turbo` 和 `speech-01-hd/turbo`。类型化请求覆盖 `task_start` 的全部字段：`voice_setting` 包含音色、语速、音量、音高、情绪、英语规范化和 LaTeX 朗读；可选 `audio_setting` 当前以带默认值的类型发送，支持 MP3、WAV、FLAC、PCM、μ-law 和 Opus；还支持发音字典、音色混合、声音效果器、字幕、语言增强和 `continuous_sound`。

客户端会在连接前验证模型与参数组合：`fluent` / `whisper` 情绪仅适用于 `speech-2.6-hd/turbo`；`continuous_sound` 仅适用于 `speech-2.8-hd/turbo`；`voice_modify` 仅对流式 MP3 文档化；音色混合最多 4 个，每个权重为 1–100，并且混合时 `voice_setting.voice_id` 必须留空。`latex_read` 只支持中文；若同时传入 `language_boost`，其值必须为 `Chinese`。不传该字段时，客户端保持省略，符合 MiniMax 文档所述由服务端设为中文。speech-01/02 不支持 Persian、Filipino 或 Tamil `language_boost`。音频采样率可为 8、16、22.05、24、32 或 44.1 kHz，声道为 1 或 2；MP3 码率可为 32、64、128 或 256 kbit/s，μ-law 格式要求 8 kHz。默认音频设置为 32 kHz、128 kbit/s、单声道。`emotion` 取值为 `happy`、`sad`、`angry`、`fearful`、`disgusted`、`surprised`、`calm`、`fluent` 或 `whisper`；`voice_modify.sound_effects` 支持 `spacious_echo`、`auditorium_echo`、`lofi_telephone` 和 `robotic`，字幕粒度支持 `sentence`、`word` 与 `word_streaming`。每个 `task_continue` 发送一段文本。音频帧会把 `data.audio` 从十六进制解码为字节，并保留完整原生事件、`extra_info`、会话 ID、trace ID 和 `is_final` 标记；Opus chunk 必须按到达顺序拼接后再解码。

调用方可在接收另一半事件流时排队发送多段文本，队列长度由本地配置限制。默认本地限制为每段 16 KiB、最多 8 段待处理文本、每条 WebSocket 消息 1 MiB、每个会话解码音频 64 MiB。这些是客户端边界，不代表 MiniMax 服务限制。`task_finish` 要求 MiniMax 完成队列并结束任务；继续读取事件直到 `TaskFinished`。

服务不会自动重连或重放文本。如果 `task_continue` 或 `task_finish` 可能已到达 MiniMax 后发送失败，结果为 `OutcomeUnknown`；不要自动重复提交。若在 `task_finished` 前断开，调用方收到 `Interrupted`，此前已交付的音频 chunk 仍由调用方持有。丢弃 session 两端会释放注入的 transport；需要显式取消时调用 `abort` 发送 WebSocket close。

契约依据 MiniMax 的[国际版普通 T2A WebSocket 文档](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket)和[大陆普通 T2A WebSocket 文档](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket)。两页分别明确给出 `.io` 与 `.cn` 的完整 WSS endpoint，并记录 Bearer 认证、`connected_success` / `task_started` 握手、`task_continue` / `task_finish`、十六进制音频、`is_final`、队列行为和终止事件。这里实现的是普通单向 T2A WebSocket，不是单独的 Bidi WebSocket 协议。
