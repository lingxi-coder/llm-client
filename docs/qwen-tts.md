# Qwen HTTP 语音合成与 SSE 音频输出

`QwenTtsService` 向百炼 `qwen3-tts-flash` 一次提交完整文本。`synthesize()` 返回短时效音频 URL；`synthesize_stream()` 返回原生 SSE 音频增量和最终 URL。两种模式都不会下载音频、自动重试或切换账户。HTTP SSE 只对输出做流式传输；独立 Qwen TTS Realtime 的双向文本输入使用另一套协议。

```rust,no_run
use lingxi_llm_client::{
    HttpTransport, RequestOptions,
    protocol::Secret,
    qwen_tts::{
        QwenTtsLanguage, QwenTtsRegion, QwenTtsRequest, QwenTtsScope, QwenTtsService,
    },
};

async fn synthesize() -> Result<(), Box<dyn std::error::Error>> {
    let transport = HttpTransport::new()?;
    let scope = QwenTtsScope::new(
        "qwen-main",
        "account-42",
        QwenTtsRegion::Singapore,
        "my-workspace-id",
    )?;
    let service = QwenTtsService::new(&transport, scope)?;
    let options = RequestOptions {
        credential: Some(Secret::new("Singapore Model Studio API key".into())),
        account_scope: Some("account-42".into()),
        ..Default::default()
    };
    let request = QwenTtsRequest::new("你好，欢迎使用语音合成。", "Cherry")
        .with_language_type(QwenTtsLanguage::Chinese);

    let result = service.synthesize(&request, &options).await?;
    println!("audio ID: {:?}", result.audio_id);
    println!("temporary audio URL: {}", result.audio_url());
    Ok(())
}
```

`QwenTtsScope` 固定 profile、账户、地域和 workspace。每次请求都要传入该地域对应的 API Key；若 `RequestOptions::account_scope` 已设置，它必须与 scope 一致。地域 DashScope 主机分别是北京 `https://dashscope.aliyuncs.com` 和新加坡 `https://dashscope-intl.aliyuncs.com`，本模块会拼接 Qwen-TTS 路径 `/api/v1/services/aigc/multimodal-generation/generation`。Qwen-TTS API 页面有一段标注为新加坡的示例，但仍使用北京主机。本实现按 Model Studio 的地域 endpoint 与 API Key 表选择路由，确保新加坡 Key 发往新加坡主机。该模型系列的合成指南没有定义 workspace 专属路由，因此 workspace ID 作为调用方账户作用域的一部分保存，但不拼进 URL。另一套 Qwen-Audio-TTS `/SpeechSynthesizer` 接口使用不同模型和 wire 契约。

`QwenTtsRequest` 发送完整文本、必填音色名称和可选 `language_type`。当前文档支持的语言值为 `Auto`、Chinese、English、German、Italian、Portuguese、Spanish、Japanese、Korean、French 和 Russian。文本最多 600 个 Unicode 字符，与当前 Qwen3-TTS-Flash API 限制一致。本切片不会把其他模型系列或指令控制字段静默加入请求。

一次性结果和 SSE 完成事件中的 `QwenTtsSynthesis` 均保留 request ID、audio ID、`finish_reason`、字符用量、过期时间和 provider 原始音频 URL。非流式响应中的音频 URL 有效期为 24 小时，文档说明此模式下 `audio.data` 为空。`Debug` 会隐藏合成文本和签名 URL。调用方可以在有效期内通过自有 HTTP 客户端下载音频；本服务不会跟随该 URL。

每次只发送一个合成请求。`QwenTtsError::dispatch_outcome()` 区分本地 `NotSent`、除 408 外的明确 HTTP 4xx `Rejected`、传输或 HTTP 408/5xx 导致的 `Unknown`，以及 HTTP 成功但响应无效的 `Accepted`。SSE 握手成功后的断流和原生供应商错误也归为 `Accepted`；`StreamProvider` 保留完整错误 JSON、用量和扩展字段，但不会通过 `Debug` 输出原文。HTTP 错误正文缺少 request ID 时，保留 `x-request-id` header。`Unknown` 和 `Accepted` 都可能已经计费，不要直接重放。本切片使用 mock transport 做契约测试，未调用真实账户。

## 完整文本的流式音频输出

`synthesize_stream()` 使用与 `synthesize()` 相同的模型、完整文本、音色、语言、地域和账户预检，添加 `X-DashScope-SSE: enable` 与 `Accept: text/event-stream`。不会向正文添加 OpenAI 风格的 `stream` 字段。

```rust,no_run
use lingxi_llm_client::{
    RequestOptions,
    qwen_tts::{QwenTtsRequest, QwenTtsService, QwenTtsStreamEventKind},
};

async fn stream_audio(
    service: &QwenTtsService<'_>,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = QwenTtsRequest::new("Hello, welcome to speech synthesis.", "Cherry");
    let mut stream = service.synthesize_stream(&request, options).await?;
    while let Some(event) = stream.next_event().await? {
        match event.kind {
            QwenTtsStreamEventKind::AudioDelta { data } => {
                // Send these PCM16 little-endian, 24 kHz, mono bytes to
                // the host's playback or storage pipeline in arrival order.
                println!("received {} audio bytes", data.len());
            }
            QwenTtsStreamEventKind::Metadata => {}
            QwenTtsStreamEventKind::Completed { synthesis } => {
                println!("complete audio URL: {}", synthesis.audio_url());
                println!("character usage: {:?}", synthesis.characters);
            }
        }
    }
    Ok(())
}
```

每个事件的 `native` 字段保留原始 JSON，包括 request/audio ID、完整用量和供应商扩展。音频增量从 `output.audio.data` 做 Base64 解码，返回 PCM16 小端、24 kHz、单声道字节；客户端不添加 WAV 文件头，也不累计完整音频。

官方完成信号是 `output.finish_reason: "stop"`；`null` 表示仍在生成。文档规定最终块的 `output.audio.data` 为空，并附完整音频 URL。URL 本身不代表完成。只有有效 stop 块之后读到干净 HTTP EOF，才产生 `Completed`；最终块后的供应商错误、额外数据、无效载荷和网络错误不会被提前成功掩盖。没有 stop 的提前 EOF 是断流。错误会终止并释放响应体，后续读取返回 `None`；干净 EOF 下仍会解析缺少最后空行的 SSE 事件。

单个 SSE 事件限制为 8 MiB，完整 wire 响应限制为 64 MiB，按 64 KiB 切片解析以限制待处理事件队列。这些是客户端资源上限，不是供应商公开额度。设置的 `RequestOptions::total_timeout` 同时约束响应体读取和最终 EOF 等待。丢弃 stream 会取消读取，不重试，也不承诺供应商停止计费。此模式不包含音频设备、双向文本 WebSocket、Qwen-Audio-TTS 或 CosyVoice。

Official references: [Qwen-TTS API reference](https://help.aliyun.com/en/model-studio/qwen-tts-api), [non-real-time speech synthesis guide](https://help.aliyun.com/en/model-studio/non-realtime-tts-user-guide), [Model Studio regions and endpoints](https://help.aliyun.com/en/model-studio/regions/), [Base URL by region](https://help.aliyun.com/en/model-studio/base-url), and [Qwen3-TTS-Flash model information](https://help.aliyun.com/en/model-studio/qwen3-tts-flash).
