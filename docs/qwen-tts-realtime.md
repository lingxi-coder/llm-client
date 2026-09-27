# Qwen 独立 TTS Realtime

`qwen_tts_realtime` 实现独立 Qwen-TTS Realtime 原生 WebSocket 协议：增量文本输入、流式音频输出、文本缓冲提交/清除，以及正常结束会话。它复用可注入的 `RealtimeTransport`；可选 `realtime-websocket` 特性提供 `RustlsWebSocketTransport`。宿主并行驱动输入与事件读取，客户端不启动后台 driver，不自动重连、重放文本、播放音频或刷新凭证。

显式选择地域：

| 地域 | 加入 model 查询参数前的 endpoint |
| --- | --- |
| 北京 | `wss://dashscope.aliyuncs.com/api-ws/v1/realtime` |
| 新加坡 | `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime` |

服务添加 `?model=...`、`Authorization: Bearer ...` 和 `X-DashScope-WorkSpace`。`QwenTtsRealtimeConfig` 绑定 profile、非秘密账户标识、地域和 workspace；`scope()` 还提供 endpoint 指纹。每次连接通过 `RequestOptions` 提供 API Key；其中的 `account_scope` 若已设置，必须与服务匹配。API Key 归属的账户与地域由宿主保证，不从 Chat 配置推断。[官方连接指南](https://help.aliyun.com/en/model-studio/interactive-process-of-qwen-tts-realtime-synthesis)。

## 模型、音色与配置

仅接受官方当前列出的精确模型 ID，未知模型在握手前拒绝。

| 系列 | 模型 ID | 地域 / 音色来源 |
| --- | --- | --- |
| Flash | `qwen3-tts-flash-realtime`、`qwen3-tts-flash-realtime-2025-11-27`、`qwen3-tts-flash-realtime-2025-09-18` | 两地 / `System` |
| Instruct | `qwen3-tts-instruct-flash-realtime`、`qwen3-tts-instruct-flash-realtime-2026-01-22` | 两地 / `System` |
| 声音复刻 | `qwen3-tts-vc-realtime-2026-01-15`、`qwen3-tts-vc-realtime-2025-11-27` | 两地 / `Cloned` |
| 声音设计 | `qwen3-tts-vd-realtime-2026-01-15`、`qwen3-tts-vd-realtime-2025-12-16` | 两地 / `Designed` |
| 旧版 | `qwen-tts-realtime`、`qwen-tts-realtime-latest`、`qwen-tts-realtime-2025-07-15` | 仅北京 / `System` |

`QwenTtsRealtimeVoice` 显式区分系统、复刻和设计音色；客户端检查来源与模型系列是否匹配。具体音色 ID、音色支持的语言和账户权限仍由供应商验证；本服务不创建定制音色。[模型与地域列表](https://www.alibabacloud.com/help/en/model-studio/realtime-tts-user-guide)、[Flash 模型卡](https://help.aliyun.com/en/model-studio/qwen3-tts-flash-realtime)、[Instruct 模型卡](https://help.aliyun.com/en/model-studio/qwen3-tts-instruct-flash-realtime)。

请求默认采用 `ServerCommit`、语言 `Auto`、24 kHz PCM。Qwen3 系列支持 PCM/WAV/MP3/Opus、8/16/24/48 kHz、0.5–2.0 倍语速与音高、0–100 音量，以及仅 Opus 可用的 6–510 kbps 比特率。旧版只接受 24 kHz PCM，不支持这些可选的速度、音量、音高或比特率控制。`instructions` 与 `optimize_instructions` 仅限 Instruct；供应商校验指令的 1,600 token 上限及中英文限制，客户端不拿字符数冒充 token 数。[客户端事件参考](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-client-events)。

## 并发输入与输出

```rust,no_run
use std::sync::Arc;
use lingxi_llm_client::{
    RequestOptions,
    realtime::RealtimeTransport,
    qwen_tts_realtime::{
        QwenTtsRealtimeConfig, QwenTtsRealtimeError, QwenTtsRealtimeEventKind,
        QwenTtsRealtimeLimits, QwenTtsRealtimeRegion, QwenTtsRealtimeRequest,
        QwenTtsRealtimeService, QwenTtsRealtimeVoice,
    },
};

async fn speak(
    transport: Arc<dyn RealtimeTransport>,
    options: &RequestOptions,
) -> Result<(), QwenTtsRealtimeError> {
    let config = QwenTtsRealtimeConfig::new(
        "qwen-intl", "billing-account-1", QwenTtsRealtimeRegion::Singapore,
        "workspace-id",
    );
    let service = QwenTtsRealtimeService::new(transport, config)?;
    let request = QwenTtsRealtimeRequest::new(
        "qwen3-tts-flash-realtime", QwenTtsRealtimeVoice::System("Cherry".into()),
    );
    let session = service.connect(options, &request, QwenTtsRealtimeLimits::default()).await?;
    let (mut input, mut events) = session.into_parts();
    // 借用两个句柄，保持 input 存活直到音频读取完成。
    let write = async {
        input.append_text("你好。").await?;
        input.append_text("这是第二段到达的文本。").await?;
        input.finish().await
    };
    let read = async {
        while let Some(event) = events.next().await? {
            match event.kind {
                QwenTtsRealtimeEventKind::AudioDelta { data, .. } => {
                    // 按协商好的音频格式交给宿主播放或保存。
                    println!("{} audio bytes", data.len());
                }
                QwenTtsRealtimeEventKind::ResponseDone { status, .. } => {
                    println!("response status: {status}");
                    // event.native 保留完整 response 和 usage。
                }
                QwenTtsRealtimeEventKind::SessionFinished => println!("session complete"),
                _ => {}
            }
        }
        Ok::<(), QwenTtsRealtimeError>(())
    };
    futures::try_join!(write, read)?;
    Ok(())
}
```

`connect()` 等待 `session.created`，发送 `session.update`，再等待 `session.updated`，之后才暴露可写会话。确认时核对模型及其官方等价别名、session ID、音色、mode、输出格式和采样率。若返回 `language_type`，它也必须与请求匹配；省略这一可选回显字段不会被拒绝。初始化事件保存在 `metadata()`，不会重复出现在普通音频事件流里。每个 JSON 客户端事件在发送前生成新的 UUID v4。

`append_text()` 对应一次 `input_text_buffer.append`。`Commit` 模式须显式 `commit()` 触发合成，两种模式下的已知空缓冲（尚未 append 或明确 clear 后）都会在本地拒绝；`ServerCommit` 模式也允许显式 commit。服务端自动分段与本地追加没有确认关联，追加后的缓冲可能已被服务端消耗，因此之后是否为空仍由供应商裁定。`commit()` 与 `clear_buffer()` 在发送完成后返回；宿主必须通过 `events.next()` 消费相应确认，后续输入才会解除 `ControlPending`。因此写入时需要持续读取事件。

`clear_buffer()` 只清除尚未提交的文本，不取消已经生成的 response，也不丢弃其后续音频。此协议不发送 Omni 的 `response.cancel`。`abort()` 仅在本地关闭，不承诺供应商停止计费。`ping()` 只委托传输层发送 WebSocket Ping，限制 125 字节，不创建 JSON ping 事件或自动保活计时器。

## 完成与失败

音频从 `response.audio.delta` 的 Base64 数据解码，并保留 response/item ID。`AudioDone` 只结束一个音频部分；`ResponseDone` 保留供应商状态和原始用量。failed、incomplete 或未知终态不会被改写为成功，本客户端会终止该会话。所有事件保留 `event.native`；无效帧和供应商错误原文保留在对应错误类型中。Debug 隐藏原始载荷和音频。

`finish()` 停止接收新文本并发送 `session.finish`。只有发送已确认成功、收到 `session.finished`、所有已观察到的 response 已结束、且 WebSocket 干净 EOF，才输出 `SessionFinished`。完成标记之后的额外数据、供应商错误、传输错误或提前 EOF 均不能视作成功。取消发送 future 会使会话关闭并唤醒阻塞的接收端，不重放任何帧。在终态清理期间取消 `events.next()` 不会丢失待交付的事件或错误；下一次调用仍会返回该结果一次。等待 finish 发送确认时也会记住已收到的 EOF，恢复被取消的读取不会再次轮询已经结束的自定义 stream。丢弃任一会话端会取消仍活动的会话，因此读取尾部输出期间应保持 input 句柄存活。[服务端事件参考](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-server-events)。

默认本地上限：单帧 4 MiB、累计解码音频 512 MiB、每次文本/指令输入 64 KiB、16 个额外初始化事件、64 个同时活动的 response。每次发送限时 30 秒，finish 确认与 EOF 共限时 60 秒；连接初始化默认限时 30 秒。它们是客户端资源约束，不是供应商额度。宿主管理调度和播放；自定义 transport 需遵守现有连接释放契约。mock 测试不能代替真实账户验收。

Qwen-Audio-TTS/CosyVoice 的 `/api-ws/v1/inference`、独立 ASR、创建音色和 Qwen-Omni 均为其他独立能力。
