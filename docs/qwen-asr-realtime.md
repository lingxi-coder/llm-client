# Qwen 独立 ASR Realtime

`qwen_asr_realtime` 实现 Qwen3-ASR 独立 WebSocket 音频转文字协议，只复用其他实时服务的可注入 `RealtimeTransport`。宿主并行驱动输入与输出；客户端不录音、不转码、不自动重连或重发音频，也不刷新凭证。

## 地域作用域与模型

`QwenAsrRealtimeConfig` 绑定 profile、非秘密账户标识、地域及 workspace，使用对应 workspace 专属 endpoint：

| 地域 | Endpoint |
| --- | --- |
| 北京 | `wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=...` |
| 新加坡 | `wss://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=...` |

每次连接通过 `RequestOptions` 提供凭证；若设置 `account_scope`，必须与服务配置相同。握手包含 `Authorization: Bearer ...` 和 `X-DashScope-WorkSpace`；默认不启用内容检查，仅当明确配置时发送 `X-DashScope-DataInspection: enable`。API Key 的账户和地域归属由宿主保证。[连接指南](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-interaction-process)。

两地均支持 `qwen3-asr-flash-realtime`、`qwen3-asr-flash-realtime-2026-02-10`、`qwen3-asr-flash-realtime-2025-10-27`。官方将稳定 ID 对应到 2025-10-27 版本，客户端接受这种等价别名回显。未知模型在握手前拒绝。[模型卡](https://help.aliyun.com/en/model-studio/qwen3-asr-flash-realtime)、[模型与地域表](https://help.aliyun.com/en/model-studio/real-time-speech-recognition-user-guide)。

## 音频与会话设置

`QwenAsrRealtimeRequest` 默认 PCM、16 kHz 和服务端 VAD（`threshold: 0.2`、`silence_duration_ms: 800`）。格式可选 `Pcm` 或 `Opus`，采样率为 8 或 16 kHz。PCM 应为单声道、有符号 16-bit 小端采样，每次追加需包含完整采样；客户端不解析或重采样编码后的 Opus 数据。

VAD threshold 必须是 -1 到 1 的有限数值，静音持续时间为 200–6000 ms。`Manual` 编码为 `turn_detection: null`。语言提示使用 `QwenAsrLanguage`；`corpus_text` 对应 `input_audio_transcription.corpus.text`。供应商校验上下文的 10,000 token 上限，客户端字节上限仅用于约束资源。[客户端事件参考](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-client-events)。

`connect()` 消费 `session.created`，发送带 UUID v4 的 `session.update`，再等待 `session.updated`。初始化 JSON 保存在 `metadata()`。确认时核对模型、session ID 和音频格式；请求 `pcm` 时接受官方样例中的 `pcm16` 回显。可选回显的采样率、VAD、语言和上下文不能与请求矛盾。官方服务端页面有一段误用了 GPT 模型名的示例；客户端不会将该错误示例视为有效 Qwen 模型确认。

## 并发使用

```rust,no_run
use std::sync::Arc;
use lingxi_llm_client::{
    RequestOptions,
    realtime::RealtimeTransport,
    qwen_asr_realtime::{
        QwenAsrRealtimeConfig, QwenAsrRealtimeError, QwenAsrRealtimeEventKind,
        QwenAsrRealtimeLimits, QwenAsrRealtimeRegion, QwenAsrRealtimeRequest,
        QwenAsrRealtimeService,
    },
};

async fn recognize(
    transport: Arc<dyn RealtimeTransport>,
    options: &RequestOptions,
    pcm16_mono_16khz: &[u8],
) -> Result<(), QwenAsrRealtimeError> {
    let config = QwenAsrRealtimeConfig::new(
        "qwen-intl", "billing-account-1", QwenAsrRealtimeRegion::Singapore,
        "workspace-id",
    );
    let service = QwenAsrRealtimeService::new(transport, config)?;
    let request = QwenAsrRealtimeRequest::new("qwen3-asr-flash-realtime");
    let session = service.connect(options, &request, QwenAsrRealtimeLimits::default()).await?;
    let (mut input, mut events) = session.into_parts();
    let write = async {
        for chunk in pcm16_mono_16khz.chunks(3200) {
            input.append_audio(chunk).await?;
        }
        input.finish().await
    };
    let read = async {
        while let Some(event) = events.next().await? {
            match event.kind {
                QwenAsrRealtimeEventKind::PartialTranscript { text, stash, .. } => {
                    println!("current preview: {text}{stash}");
                }
                QwenAsrRealtimeEventKind::TranscriptCompleted { transcript, .. } => {
                    println!("final transcript: {transcript}");
                }
                QwenAsrRealtimeEventKind::TranscriptFailed { code, .. } => {
                    println!("recognition failed: {code:?}");
                }
                QwenAsrRealtimeEventKind::SessionFinished => println!("session complete"),
                _ => {}
            }
        }
        Ok::<(), QwenAsrRealtimeError>(())
    };
    // 两个句柄都保持存活，直到最终事件已读完。
    futures::try_join!(write, read)?;
    Ok(())
}
```

VAD 模式持续追加音频，输入结束后调用 finish；显式 `commit()` 会在发送前被拒绝。Manual 模式追加一段完整语音后调用 commit，空缓冲本地拒绝。再次追加或提交前，需要读取 commit 确认；已经成功发送的 commit 可以紧接 finish，无须先读取确认，这与官方流程一致。Manual 会话有尚未提交的本地音频时，必须先 commit 再 finish。

当前独立 ASR 客户端/服务端参考没有定义 `input_audio_buffer.clear`，因此不暴露 clear 方法，也不借用 Omni 或 TTS 的事件。`abort()` 只在本地关闭；`ping()` 是明确的传输层 WebSocket Ping，不是供应商 JSON 事件，也不启动自动保活。

## 转写与完成语义

`PartialTranscript.text` 是当前已确认前缀，`stash` 是可修订后缀；使用两者拼接结果替换当前 item 的预览。它们不是可不断追加的 delta，stash 为空也不表示 item 已最终完成。`TranscriptCompleted.transcript` 才是最终识别结果，并保留 item ID、content index、语言、情绪和原始载荷。`conversation.item.created` 中的 item status 为 `completed` 也不等于转写完成。此模型不返回词级时间戳；语音开始/停止偏移是独立的 VAD 事件。[服务端事件参考](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-server-events)、[识别指南](https://help.aliyun.com/en/model-studio/real-time-speech-recognition-user-guide)。

所有事件 JSON 及供应商用量、扩展字段都保存在 `event.native`。供应商 `error` 保留原始错误载荷；单 item 识别失败返回 `TranscriptFailed`，只移除该 item 的待完成状态；其他 item 和后续输入可继续，客户端不会重发失败音频。通用供应商错误、传输错误和无效生命周期事件仍会终止会话。`SessionFinished` 表示会话结果已全部接收，不代表所有 item 都识别成功；调用方需单独记录失败 item。Debug 隐藏原始载荷和转写文本。

ASR 要求**收到 `session.finished` 后由客户端主动关闭**。接收端等待 finish 发送确认，再发起 WebSocket close，同时继续读取服务端事件。只有供应商完成标记、所有已观察 item 均已结束、本地 close 确认与干净 EOF 齐备，才返回 `SessionFinished`。无语音的会话可以直接结束而没有转写。晚到的供应商/网络错误、提前 EOF 都不会被当成成功。

close future 会跨 `events.next()` 的取消保留，且等待时不持有共享 sink 锁，接收端可继续推进。待交付的终态事件/错误和已观察 EOF 也不会因取消读取而丢失。取消音频或控制发送会使会话关闭并唤醒读端，不重发。读取最终结果期间需保持 input 句柄存活；丢弃任一端会取消活动会话。

默认客户端上限：单帧 4 MiB、每次原始音频追加 1 MiB、每会话输入音频 512 MiB、每个文本事件或上下文 64 KiB、累计转写文字 16 MiB、16 个额外初始化事件、64 个活动 item。每次发送默认限时 30 秒，最终识别与关闭合计 60 秒；初始化连接默认 30 秒，并受已提供的 `RequestOptions::total_timeout` 进一步限制。这些是本地资源上限，不是供应商额度。mock 测试不代表真实账户验收。
