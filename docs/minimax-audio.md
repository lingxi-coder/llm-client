# MiniMax 语音识别

[English](minimax-audio.en.md)

`client.minimax_audio(endpoint).transcribe(...)` 是 MiniMax ASR 1.0 的独立同步文件转写服务。它不依赖 Chat 的 `base_url`，也不复用 OpenAI 的转写字段。调用方必须显式选择完整区域 endpoint，并在每次请求的 `RequestOptions::credential` 中传入对应账户的 API Key：

- 国际 endpoint：`https://api.minimax.io/v1/speech_to_text`
- 中国大陆 endpoint：`https://api.minimax.cn/v1/speech_to_text`

服务只接受这两个 HTTPS 主机上的 `/v1/speech_to_text` 路径，不接受 URL 用户信息、查询、片段或非 443 端口。端点选择由调用方负责；服务不会从聊天路由推断 endpoint、切换区域或 failover。ASR 是无状态调用，不保留账户作用域、凭据或缓存数据。

```rust,no_run
use lingxi_llm_client::{
    audio::AudioInput,
    minimax_audio::{
        MiniMaxAudioService, MiniMaxSpeechLanguage, MiniMaxTranscriptionRequest,
        MINIMAX_ASR_INTERNATIONAL_ENDPOINT,
    },
    protocol::Secret,
    HttpTransport, RequestOptions,
};

async fn transcribe() -> Result<(), Box<dyn std::error::Error>> {
    let transport = HttpTransport::new()?;
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT)?;
    let input = AudioInput::from_bytes("meeting.wav", "audio/wav", b"audio bytes".to_vec());
    let request = MiniMaxTranscriptionRequest {
        language: Some(MiniMaxSpeechLanguage::Chinese),
        ..Default::default()
    };
    let options = RequestOptions {
        credential: Some(Secret::new("account API key".into())),
        ..Default::default()
    };
    let result = service.transcribe(input, &request, &options).await?;
    println!("{}", result.text);
    Ok(())
}
```

已有的 `LlmClient` 或 `ClientSnapshot` 也可通过 `client.minimax_audio(endpoint)` 创建此服务；两者都要求显式提供 endpoint。不要从 MiniMax Chat 的 `base_url` 推导语音 endpoint。

`AudioInput` 是带声明长度的一次性字节流。MiniMax 接受 WAV、AIFF、FLAC、M4A/ALAC、MP3、AAC、Opus 和 Ogg 容器，文件不超过 50 MB、时长不超过 500 秒；不支持裸 PCM。客户端会在发送 multipart 请求时检查大小、MIME 类型、文件扩展名和实际字节数。通用字节流无法提供录音时长，因此超过 500 秒的录音可能由 MiniMax 拒绝。

`MiniMaxTranscriptionRequest` 可选择 `json`（默认）、`verbose_json`、SRT 或 VTT 输出，句子级或词级时间戳，以及可选的强类型语言提示。省略语言提示时启用混合语言识别。JSON 响应保留原生 provider JSON、文本、时长、说话人数、带说话人信息的片段和 trace ID（若 provider 返回）。SRT/VTT 响应保留完整字幕文本。`verbose_json`、SRT 和 VTT 会启用 provider 的说话人分离与强制对齐。

`transcribe()` 发送 `stream=false` 并收集完整响应。增量式 JSON 转写可使用 `transcribe_stream()`：

```rust,no_run
use futures::StreamExt;

async fn transcribe_incrementally(
    service: &lingxi_llm_client::minimax_audio::MiniMaxAudioService<'_>,
    input: lingxi_llm_client::audio::AudioInput,
    options: &lingxi_llm_client::RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut events = service
        .transcribe_stream(
            input,
            &lingxi_llm_client::minimax_audio::MiniMaxTranscriptionRequest::default(),
            options,
        )
        .await?;
    let mut transcript = String::new();
    let mut duration = None;
    while let Some(event) = events.next().await {
        let event = event?;
        transcript.push_str(&event.delta);
        if event.finish {
            duration = event.duration_seconds;
        }
    }
    println!("{transcript} ({duration:?} seconds)");
    Ok(())
}
```

`transcribe_stream()` 设置 `stream=true`、请求 `text/event-stream`，并且只支持 `response_format=json`。解析器处理任意传输分块，要求事件 index 从 0 开始并逐一递增，且终止的 `finish=true` 事件必须包含有限非负时长。调用方按顺序拼接 delta；服务在终止事件后停止读取。`verbose_json`、SRT 和 VTT 不能与流式模式组合。默认请求期限为 10 分钟，可通过 `RequestOptions::total_timeout` 修改。

上传流只消费一次；客户端不会自动重试或切换区域。`MiniMaxAudioError::dispatch()` 区分本地预检失败（`NotSent`）、明确的 HTTP 4xx 拒绝（`Rejected`）、传输或服务端结果不确定（`Unknown`），以及 HTTP 成功后响应无法解析（`Accepted`）。`Unknown` 和 `Accepted` 可能已经产生费用，不要盲目重放请求。本次没有调用真实 MiniMax 账户；接口契约由本地 mock transport 测试覆盖。

官方文档：[国际版语音识别](https://platform.minimax.io/docs/api-reference/speech-to-text)、[中国大陆语音识别](https://platform.minimax.cn/docs/api-reference/speech-to-text)。
