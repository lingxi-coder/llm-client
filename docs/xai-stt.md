# xAI 实时语音转文字

`xai_stt::XaiSttSession` 实现 xAI 固定路由 `wss://api.x.ai/v1/stt` 上的语音转写 WebSocket 协议。它复用有界实时传输层，但与 xAI 语音对话 API 的二进制音频和转写事件契约分开。宿主传入 `Arc<dyn RealtimeTransport>` 和本次连接使用的 API key。当前线路细节以 xAI 的[语音转文字文档](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text)为准。

```rust,no_run
use futures::future::join;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{RealtimeError, RealtimeLimits, RealtimeTransport},
    xai_stt::{XaiSttConfig, XaiSttEvent, XaiSttSession},
};
use std::{error::Error, sync::Arc};

async fn transcribe(
    transport: Arc<dyn RealtimeTransport>,
    api_key: String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (session, driver) = XaiSttSession::connect(
        transport,
        Secret::from(api_key),
        XaiSttConfig::default(),
        RealtimeLimits::default(),
    )
    .await?;
    let (control, mut events) = session.into_parts();

    // 真实应用应按合适的节奏发送原始 PCM16 分块。
    control.send_audio(vec![0_u8, 0])?;
    control.finish_audio()?;

    let (driver_result, receive_result) = join(driver.run(), async move {
        while let Some(event) = events.next().await {
            if matches!(event, XaiSttEvent::Completed { .. }) {
                break;
            }
        }
        Ok::<(), RealtimeError>(())
    })
    .await;
    driver_result?;
    receive_result?;
    Ok(())
}
```

`connect` 会等待 `transcript.created` 后才返回；`connect_timeout` 同时限制 WebSocket 握手和等待 ready 事件的时间。`XaiSttConfig` 可选择 `grok-voice-transcribe-1.0` 或默认的 `grok-voice-transcribe-2.0`，以及 `pcm`、`mulaw`、`alaw`、`opus` 原始编码，并配置语言格式、静音端点、说话人区分、填充词、多声道、关键词、Smart Turn 和 VAD 阈值。`sample_rate_hz: None` 使用服务端默认的 16 kHz；非 Opus 采样率可为 8000、16000、22050、24000、44100 或 48000 Hz。Opus 不发送采样率参数，只支持单声道，并要求每个二进制帧恰好包含一个原始 Opus packet。

`send_audio` 将传入字节作为一个二进制 WebSocket 帧发送，不做 Base64 转换。音频采集、分块节奏和重采样由宿主负责。`finalize_utterance` 发送不指定声道的小写消息 `{"type":"finalize"}`，同时保持输入开放；这里依照文档 Client Messages 列表中的小写写法。该页面另外的按键说话示例用了 `Finalize`，因此本适配器不会假定两种大小写都有效。`finish_audio` 发送 `{"type":"audio.done"}`；只有有界队列接受了该消息后，本地输入才会关闭。运行返回的 driver，同时读取事件流。

类型化 partial 事件包含文档列出的文本、words、最终性标记、时间戳、可选声道索引和可选 Smart Turn 置信度。类型化 `TranscriptDone` 要求有限且非负的 duration，以及有效且未重复的声道索引。多声道模式下，只有 `audio.done` 已实际发到 socket，并且每个配置声道都收到一个有效 final 事件后，driver 才把服务端 EOF 作为正常结束。`Completed { expected_channels }` 表示该 EOF；事件不伪造 close code。缺失或畸形的 final、服务端错误和传输错误仍会作为错误或事件保留。无法识别或格式错误的 provider 事件会在 `ProviderEvent` 中保留原生 JSON。

此模块不会自动重连或重放音频，也不负责设备采集，更不会把语音转写流包装成语音对话。启用 crate 的 `realtime-websocket` feature 后可使用 Rustls WebSocket transport；否则传入由宿主提供的 transport 实现。
