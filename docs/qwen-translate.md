# Qwen LiveTranslate Realtime（WebSocket）

`realtime` 中的 Qwen LiveTranslate 适配器实现阿里云百炼实时语音/视频翻译 WebSocket 协议。它与 Qwen-Omni Realtime 使用独立配置和事件类型：Qwen 3.5 的增量文本包含确认的 `text` 与暂存的 `stash`，Qwen 3.8 使用追加式 `delta`。当前支持官方文档列出的北京和新加坡 workspace endpoint，以及 Qwen 3.5 stable alias、其 `2026-05-19` snapshot 和 Qwen 3.8 stable model ID。

依据：[Qwen LiveTranslate 模型指南](https://help.aliyun.com/en/model-studio/qwen3-5-livetranslate-flash-realtime)、[客户端事件](https://help.aliyun.com/en/model-studio/live-translator-client-events)、[服务端事件](https://help.aliyun.com/en/model-studio/live-translator-server-events)。

## 路由、账户与凭证

`QwenLiveTranslateRoute::new` 要求区域和单个 workspace ID，并构造该区域专用的 WebSocket endpoint。连接使用官方要求的 `model` 查询参数和 `Authorization: Bearer …` 握手 header。`QwenLiveTranslateScope` 将 profile、账户、region、workspace 和 endpoint fingerprint 绑定；API key 作为 `Secret<String>` 逐次传入，不保存在 scope 中。

```rust,no_run
use std::sync::Arc;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{
        Qwen35LiveTranslateConfig, QwenLiveTranslateConfig,
        QwenLiveTranslateRegion, QwenLiveTranslateRoute,
        QwenLiveTranslateScope, QwenLiveTranslateSession,
        RealtimeDriver, RealtimeError, RealtimeLimits, RealtimeTransport,
    },
};

async fn connect_live_translate(
    transport: Arc<dyn RealtimeTransport>,
    workspace_id: String,
    api_key: Secret<String>,
) -> Result<(QwenLiveTranslateSession, RealtimeDriver), RealtimeError> {
    let route = QwenLiveTranslateRoute::new(
        QwenLiveTranslateRegion::Beijing,
        workspace_id,
    )?;
    let scope = QwenLiveTranslateScope::new("qwen-live", "account-1", &route)?;
    let config = QwenLiveTranslateConfig::qwen35(
        Qwen35LiveTranslateConfig::default(),
    );
    QwenLiveTranslateSession::connect(
        transport,
        route,
        scope,
        api_key,
        config,
        RealtimeLimits::default(),
    )
    .await
}
```

连接等待服务端 `session.created`，再发送 `session.update`，收到 `session.updated` 后才返回。配置枚举为两个 wire schema 保持分离：`Qwen35LiveTranslateConfig` 编码 flat `modalities`、采样率、输入 PCM/Opus 与可选 `turn_detection: null`；`Qwen38LiveTranslateConfig` 编码 `output_modalities` 和嵌套 `audio.input`/`audio.output`。两种配置均能选择文本或文本加音频，不能选择 audio-only。Qwen 3.5 输入支持 8/16 kHz PCM16 mono 或 Opus；Qwen 3.8 使用 16 kHz PCM16 mono；输出音频按文档默认值报告为 PCM16 24 kHz。

## 输入与控制

- 音频使用 `RealtimeInput::Audio`，编码为 Base64 `input_audio_buffer.append`。PCM16 样本必须完整；3.5 Opus 会话只接受 `Encoded { mime_type: "audio/opus" }`。采集和分块由宿主负责。
- 图像帧使用 `RealtimeInput::Image`，只接受 JPG/JPEG。百炼文档给出 500 KB 的 Base64 前上限；客户端保守地限制为 500,000 字节。至少成功发送一个音频 append 后才能发送图像；模型文档将图像速率限制为每秒最多两张，宿主负责抽帧与限速。
- `clear_audio()` / `RealtimeInput::ClearAudio` 发送 `input_audio_buffer.clear`，清除未提交音频。
- Qwen 3.5 手动模式通过 `commit_audio()` / `RealtimeInput::CommitAudio` 提交当前音频。Server VAD 模式和 Qwen 3.8 会在本地拒绝 commit；Qwen 3.8 配置只开放文档明确列出的 speaker detection 与 server VAD。
- `finish()` / `RealtimeInput::FinishSession` 发送 `session.finish`，不会自动关闭连接。要保留最终源语言识别和翻译结果，应继续读事件，直到 `QwenLiveTranslateEvent::SessionFinished`，再显式调用 `close_after_finished()`。若会话失败或卡住，`close()` 仍允许宿主提前中止。

不接受普通文本、函数工具结果、继续生成或中断操作；这些输入不是此 LiveTranslate client-event 协议的一部分。

## 配置与事件

Qwen 3.5 配置可指定目标/源语言、源语音识别、8/16 kHz 输入、VAD 阈值和静音时长、语音、语音克隆模式，以及仅在目标语言为 `zh` 或 `en` 时生效的同语种跳过选项。术语映射是有界的本地配置：客户端最多接受 1000 项，与百炼文档“建议不超过 1000 项”一致，但该值不是服务端限制声明。启用源转写时使用固定模型 `qwen3-asr-flash-realtime`。Qwen 3.8 配置使用自己的嵌套 turn-detection schema；它总是返回源转写，且不开放 Qwen 3.5 的 same-language skip 或 voice-clone 字段。语言和语音可用性仍由百炼服务端校验。

事件枚举按模型保留语义差异：

- Qwen 3.5 使用 `TextProgress35 { text, stash }` 或 `AudioTranscriptProgress35 { text, stash }` 作为未完成片段；`TextDone` / `AudioTranscriptDone` 携带完整最终文本。
- Qwen 3.8 使用 `TextDelta38 { delta }` 或 `AudioTranscriptDelta38 { delta }`。请按到达顺序追加 `delta`；不要把它解释为 Qwen 3.5 的可替换 `text` / `stash`。
- 输入源转写在 Qwen 3.5 是 `SourceTranscription35 { text, stash, .. }`，在 Qwen 3.8 是 `SourceTranscriptionDelta38 { delta, .. }`；完成与失败分别有 typed 事件。
- `AudioDelta` 返回解码后的 PCM16 24 kHz 字节；`ResponseDone` 保留完整 native response；未映射的事件保留为 `Native { event_type, native }`。
- `error` 映射到 `ProviderError`，由宿主决定是否结束会话。传输不会自动重试或重连。

本适配器不采集/播放音频、不抽取或限速视频帧、不访问远程图片 URL，也不自动重试或重连。它没有进行真实账户请求；测试应使用注入式假 transport 验证协议。
