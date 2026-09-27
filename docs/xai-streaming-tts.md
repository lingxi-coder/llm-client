# xAI WebSocket 流式文本转语音

`xai_streaming_tts` 实现 xAI 的双向流式文本转语音 WebSocket 协议，固定连接到 `wss://api.x.ai/v1/tts`。这是文本转音频流，与 xAI 语音到语音的实时对话服务及 HTTP TTS API 相互独立。宿主提供 `RealtimeTransport`，并为每次连接传入 API key。

```rust,no_run
use futures::future::join;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{RealtimeError, RealtimeLimits, RealtimeTransport},
    xai_streaming_tts::{
        XaiStreamingTtsConfig, XaiStreamingTtsEvent, XaiStreamingTtsSession,
    },
};
use std::{error::Error, sync::Arc};

async fn synthesize_one(
    transport: Arc<dyn RealtimeTransport>,
    api_key: String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (session, driver) = XaiStreamingTtsSession::connect(
        transport,
        Secret::from(api_key),
        XaiStreamingTtsConfig::new("en"),
        RealtimeLimits::default(),
    )
    .await?;
    let (control, mut events) = session.into_parts();
    control.send_text_delta("Hello from a streaming TTS session.")?;
    control.finish_utterance()?;

    let close_control = control.clone();
    let receive = async move {
        while let Some(event) = events.next().await {
            match event {
                XaiStreamingTtsEvent::AudioDelta { audio, .. } => {
                    // 在宿主中保存这些编码后的音频字节，或继续转发。
                    let _ = audio;
                }
                XaiStreamingTtsEvent::AudioDone { .. } => {
                    close_control.close().await?;
                    return Ok(());
                }
                XaiStreamingTtsEvent::ProviderError { message, .. } => {
                    close_control.close().await?;
                    return Err(RealtimeError::Codec { message });
                }
                XaiStreamingTtsEvent::ConnectionInterrupted { message } => {
                    return Err(RealtimeError::Transport { message });
                }
                _ => {}
            }
        }
        Err(RealtimeError::UnexpectedRemoteClose)
    };

    let (driver_result, receive_result) = join(driver.run(), receive).await;
    driver_result?;
    receive_result?;
    Ok(())
}
```

WebSocket upgrade 成功即表示连接就绪；服务端不会发送 `created` 或 `ready` 事件。`XaiStreamingTtsConfig::new(language)` 会设置必需语言，其他默认值为服务端默认 voice、MP3、24 kHz、128 kbit/s、1.0 倍速和 0 级流式延迟优化。设置 `voice` 可指定内置或自定义音色。支持 MP3、WAV、PCM、μ-law 和 A-law。采样率可设为 8000、16000、22050、24000、44100 或 48000 Hz；只有 MP3 接受 `bit_rate`，取值为 32000、64000、96000、128000 或 192000 bit/s。语速范围是 0.7 到 1.5。WebSocket 指南记录的 `optimize_streaming_latency` 值为 0、1、2。`text_normalization` 和 `with_timestamps` 是可选布尔值。

使用 `send_text_delta` 发送非空文本片段；每片最多 60,000 个 Unicode 字符。调用一次 `finish_utterance` 会发送 `text.done`，之后继续读取 `AudioDelta`，直到 `AudioDone`。音频增量已从 Base64 解码为所选编码格式的原始字节。发送、接收期间需要持续运行 driver；命令队列、事件队列和单帧大小受 `RealtimeLimits` 限制。本地队列已满时，命令会返回 `QueueFull`，且不会推进本地语句状态。

一个 WebSocket 连接可以承载多轮语句。`AudioDone` 表示一轮音频响应结束，不会关闭连接。取消时调用 `clear_utterance`，并由宿主丢弃已经缓冲的音频；收到 `AudioClear` 后才能发送新文本。可通过 `update_replacements` 排入 `session.update` 替换词映射；变更在下一轮语句开始时生效，`SessionUpdated` 表示收到服务端确认。映射控制服务端短语替换，不会改写客户端发送的文本。

当文档规定的字段有效时，已知音频、清除和 session-update 事件会解码为类型化事件。未知或格式错误的事件保留在 `ProviderEvent` 中；例如，缺少有效 `trace_id` 的 `audio.done` 会原样保留，但不会完成语句。服务端 `error` 消息通过 `ProviderError` 提供。远端 EOF、传输故障和连接中断属于中断，不会视为成功完成；格式错误的事件保持原样，也不会推进会话状态。客户端不会自动重连或重放文本。调用 `close` 会在本地关闭 WebSocket；它与 `finish_utterance` 不同，也不代表合成已完成。除 Base64 传输解码以外的音频处理、缓冲、播放和设备访问由宿主负责。

协议依据 xAI 的[流式文本转语音指南](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech)及[语音推理 API 参考](https://docs.x.ai/developers/rest-api-reference/inference/voice)。WebSocket 路由使用查询参数和 JSON 文本消息，不应向其发送 HTTP 路由的 JSON body。启用 `realtime-websocket` feature 可使用内置 Rustls WebSocket transport；上面的示例使用默认 feature 下的注入式 transport API。
