# Vertex AI Gemini 语音合成

`hosting::vertex::speech` 通过 Vertex AI publisher-model REST API 调用 Gemini-TTS。它与 Gemini Developer API Interactions TTS 和 Cloud Text-to-Speech API 相互独立；调用者需要明确选择 Google Cloud 项目、地区、模型，并提供 bearer access token。

```rust,no_run
use lingxi_llm_client::{
    client::RequestOptions,
    protocol::Secret,
    transport::HttpTransport,
    hosting::vertex::speech::{
        VertexSpeechModel, VertexSpeechRequest, VertexSpeechScope, VertexSpeechService,
    },
};

async fn synthesize() -> Result<(), Box<dyn std::error::Error>> {
    let http = HttpTransport::new()?;
    let scope = VertexSpeechScope::new(
        "google-cloud",
        "team-project-account",
        "my-google-project",
        "us-central1",
    )?;
    let service = VertexSpeechService::new(&http, scope)?;
    let request = VertexSpeechRequest::single(
        VertexSpeechModel::Gemini25FlashTts,
        "Welcome to the demo.",
        "en-US",
        "Kore",
    )
    .with_prompt("Speak in a calm, clear voice");

    // Obtain and refresh Google Cloud access tokens in the host application.
    let mut options = RequestOptions::default();
    options.credential = Some(Secret::new("caller-supplied-access-token".to_owned()));
    let result = service.synthesize(&request, &options).await?;
    assert_eq!(result.pcm_sample_rate_hz, 24_000);
    Ok(())
}
```

服务会向 `projects/{project}/locations/{location}/publishers/google/models/{model}` 发送一次 `generateContent` 请求。返回值包含解码后的无 WAV 头单声道 PCM16 音频，采样率为 24 kHz，并保留 Google 原生响应 JSON。服务不会播放或写入音频、获取凭证、下载远端媒体，也不会重试结果不确定的请求。

当前 Vertex 指南列出四个模型：`gemini-3.1-flash-tts-preview`、`gemini-2.5-flash-tts`、`gemini-2.5-flash-lite-preview-tts` 和 `gemini-2.5-pro-tts`。3.1 预览模型目前列在 `global` 地区；2.5 模型还列有若干地区。适配器会在发送前检查模型与地区是否同时出现在该表中。Flash-Lite 仅支持单说话人，其余所列模型支持双说话人对话。Google 可能调整模型与地区的可用范围。

`VertexSpeechRequest::multi_speaker` 接收两位说话人及其音色声明，以及带说话人标签的文本轮次。服务会把标签格式化进 Vertex 的 `contents` 字段，并设置 `generationConfig.speechConfig.multiSpeakerVoiceConfig`。两种请求都可附加自然语言风格提示。整个 `contents` 字段最多 8,000 字节，这是 Google 文档给出的上限；超过约 655 秒的输出可能被截断。

`synthesize_stream` 使用单次请求，通过 `streamGenerateContent?alt=sse` 接收服务端流式响应。每次调用 `next_event()` 都会保留一条原生 `GenerateContentResponse`，并返回该响应中解码出的音频字节。只有候选项提供非空 `finishReason`，或 Google 返回文档所述的“无候选项但有 `promptFeedback.blockReason`”结果时，流才算正常完成；在这两种信号出现前结束会报告提前 EOF。类型化的 `finish_reason` / `completion_reason` 和原生 JSON 会保留非 `STOP` 的结束原因，供调用方判断输出是否完整。这是单次请求的服务端流式响应；服务不会分多块发送文本输入，也不会在连接中断后重连。

Google 将 Vertex Gemini-TTS 输出定义为无 WAV 头的 24 kHz PCM16。如果响应带有音频 MIME 类型，适配器接受原始线性 PCM 类型（`audio/pcm` 或 `audio/L16`），并拒绝与文档冲突的 codec、采样率、声道数或位深参数。若响应未提供 MIME 元数据，则依据 Vertex TTS 契约标注为 PCM16、24 kHz、单声道。需要容器格式时，请使用其他已文档化的 API。`account_scope` 只是本地账户身份标签；Google 认证和令牌刷新由调用方负责。如果设置了 `RequestOptions.account_scope`，其值必须与 scope 相同。

来源：[Cloud Text-to-Speech 和 Vertex AI 的 Gemini-TTS 指南](https://docs.cloud.google.com/text-to-speech/docs/gemini-tts)、[Google Gemini TTS 的 PCM MIME 文档](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation)、[Vertex `streamGenerateContent` REST 方法](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.publishers.models/streamGenerateContent)、[Vertex `GenerateContentResponse` 终止字段](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse)、[Google Gen AI SDK 流式路由实现](https://github.com/googleapis/js-genai/blob/main/src/models.ts#L3634-L3717)。
