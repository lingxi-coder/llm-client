# Gemini 原生语音合成

`gemini_speech` 封装 Google Gemini Developer API 的 unary 和 SSE 文字转语音能力。当前 TTS 指南使用 Interactions API：

```text
POST https://generativelanguage.googleapis.com/v1beta/interactions
```

它不是 Generate Content 的音频生成路径，也不是面向实时对话的 Live API。此适配器覆盖 Gemini 3.8 Flash TTS 和 Gemini 3.8 Flash-Lite TTS 的**单说话人与多说话人 unary 及流式**合成。

服务绑定显式 profile、Google Cloud 项目/账户范围和 Google 官方 Interactions endpoint。每次 `synthesize` 都单独传入 `&Secret<String>`；服务不保存 API key。传输错误可能发生在 Google 已接收请求后，调用结果会标记为未知，且不会自动重试。

## 请求与结果

请求正文遵循 TTS 指南的 Interactions JSON：`model`、带文本和可选 `speech_metadata.style` 注释的 `input`、`response_format` 和 `generation_config.speech_config`。unary 请求发送 `stream: false`，流式请求发送 `stream: true`。调用方必须选择已发布的模型、音色、格式和采样率；模型与格式使用枚举，避免发送未支持的值。

支持的输出 MIME 类型为 `audio/wav`、`audio/l16`、`audio/mulaw` 和 `audio/alaw`；适配器允许官方指南举例的 8 kHz、16 kHz 和 24 kHz 采样率。默认 unary `audio/wav` 是带 RIFF 头的 WAV；默认 streaming `audio/l16` 是无头的 16-bit little-endian PCM。

Google 在完成的 Interaction `steps[].content[]` 中返回 base64 音频块。服务按官方字段解码出 `audio` 字节，并在 `native` 中保留完整 JSON 响应、用量、Interaction ID 和提供方原始音频块。丢失音频块或无效 base64 会作为已接受但响应无效处理。

```rust,ignore
use lingxi_llm_client::{
    providers::google::speech::{
        GeminiSpeechFormat, GeminiSpeechModel, GeminiSpeechSampleRate,
        GeminiSpeechScope, GeminiSpeechService, GeminiSpeechRequest,
        GEMINI_SPEECH_ENDPOINT,
    },
    protocol::Secret,
    transport::HttpTransport,
};

let http = HttpTransport::new()?;
let scope = GeminiSpeechScope::new(
    "gemini-tts",
    "google-project-prod",
    GEMINI_SPEECH_ENDPOINT,
)?;
let speech = GeminiSpeechService::new(&http, scope)?;
let request = GeminiSpeechRequest::new(
    GeminiSpeechModel::Gemini38FlashTts,
    "Have a wonderful day!",
    "Kore",
)?
.with_style("cheerful and friendly")?
.with_format(GeminiSpeechFormat::Wav)
.with_sample_rate(GeminiSpeechSampleRate::Hz24000);
let key = Secret::new(api_key);
let result = speech.synthesize(&key, &request).await?;
audio_sink.write_all(&result.audio).await?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

双人对话可使用 `GeminiSpeechSpeaker` 为两个 speaker label 配置预置音色，并使用 `GeminiSpeechTurn` 分别提供每段文本。每段可以有自己的 style。请求会发送 `speech_config.mode: "conversational"`；每段都必须引用已配置的 speaker。该接口遵循 Google 单次多说话人请求最多两人的限制，并要求使用预置音色。设计或复刻的自定义音色需要按说话人分别调用合成。

```rust,ignore
use lingxi_llm_client::providers::google::speech::{
    GeminiSpeechModel, GeminiSpeechRequest, GeminiSpeechSpeaker, GeminiSpeechTurn,
};

let request = GeminiSpeechRequest::new_multi_speaker(
    GeminiSpeechModel::Gemini38FlashTts,
    vec![
        GeminiSpeechSpeaker::new("Joe", "Puck"),
        GeminiSpeechSpeaker::new("Jane", "Kore"),
    ],
    vec![
        GeminiSpeechTurn::new("Joe", "How's it going today Jane?")
            .with_style("cheerful and friendly"),
        GeminiSpeechTurn::new("Jane", "Not too bad, how about you?")
            .with_style("calm and relaxed"),
    ],
)?;
let result = speech.synthesize(&key, &request).await?;
audio_sink.write_all(&result.audio).await?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## 流式合成

Interactions 流返回 SSE。TTS 的 `step.delta` 事件在 `delta.type == "audio"` 时把 base64 音频放在 `delta.data`；服务将每个增量解码为 `audio_chunk`，同时把完整原生事件保留在 `native`。3.8 TTS 的流式默认是无 WAV 头的 `audio/l16`，24 kHz 单声道 PCM。单说话人可用 `GeminiSpeechRequest::new_streaming`，多说话人可用 `new_multi_speaker_streaming` 明确此默认；也可以选择官方支持的其他格式和采样率。

```rust,ignore
use lingxi_llm_client::{
    providers::google::speech::{GeminiSpeechModel, GeminiSpeechRequest, GeminiSpeechScope, GeminiSpeechService, GEMINI_SPEECH_ENDPOINT},
    protocol::Secret,
    transport::HttpTransport,
};

let http = HttpTransport::new()?;
let scope = GeminiSpeechScope::new("gemini-tts", "google-project-prod", GEMINI_SPEECH_ENDPOINT)?;
let speech = GeminiSpeechService::new(&http, scope)?;
let request = GeminiSpeechRequest::new_streaming(
    GeminiSpeechModel::Gemini38FlashTts,
    "Have a wonderful day!",
    "Kore",
)?;
let key = Secret::new(api_key);
let mut stream = speech.synthesize_stream(&key, &request).await?;
while let Some(event) = stream.next_event().await? {
    if let Some(audio_chunk) = &event.audio_chunk {
        audio_sink.write_all(audio_chunk).await?;
    }
    // event.native retains the original event JSON; event.event_type identifies its lifecycle step.
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

流不在内存中拼接完整音频。它限制单条 SSE 事件为 8 MiB、单次响应线缆数据为 96 MiB、累计解码音频为 64 MiB；总体读取仍受 300 秒请求 deadline 限制。`interaction.completed` 后还要求官方 `done` / `[DONE]` 标记，缺少标记、传输中断、Provider `error` 事件、格式错误数据和超限都会显式报错。中断错误保留 Interaction ID、账户/endpoint-scoped 引用、最新 `event_id` 游标以及已交付的音频字节数。服务不自动重连或重试；如果该 Interaction 可通过 Google API 读取，调用方可以依据官方 GET `stream=true&last_event_id=...` 契约自行决定是否续接。

## 当前边界

此服务不管理 voice design / voice replication 资源。使用自定义音色时，应通过 Google Voices API 单独创建并管理，然后将返回的语音 ID传给单说话人请求。Google 的 Live API 面向双向实时语音交互，应使用独立的 Live 服务。

文本和请求体有本地容量上限，以防单次调用无界占用内存；模型输入 token 限制仍由 Google 执行。模拟传输测试不会发出真实 API 请求。

## 官方资料

- [Gemini Text-to-speech generation](https://ai.google.dev/gemini-api/docs/speech-generation)：TTS 模型、Interactions 请求、音色、输出格式和限制。
- [Streaming interactions](https://ai.google.dev/gemini-api/docs/streaming)：SSE 事件序列、`interaction.completed`、`error` 与 `[DONE]`。
- [Gemini Interactions API](https://ai.google.dev/api/interactions-api)：Interactions REST 资源与原生 response steps。
- [Gemini API release notes](https://ai.google.dev/gemini-api/docs/changelog)：Gemini 3.8 TTS GA 模型 ID。
- [Gemini Voices API](https://ai.google.dev/api/voices)：预置音色、扩展音色库和自定义音色资源管理。
