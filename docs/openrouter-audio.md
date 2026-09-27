# OpenRouter 音频服务

`openrouter_audio` 提供 OpenRouter 专属的语音转文字（STT）与文字转语音（TTS）接口。它只使用 OpenRouter 的 `/api/v1/audio/transcriptions` 和 `/api/v1/audio/speech` 路由，不从 OpenAI 服务或模型目录推导音频能力。每次调用都通过 `RequestOptions` 显式接收 OpenRouter bearer key 和可选账户范围；账户范围随结果返回，不会发送到服务端。

```rust,ignore
use bytes::Bytes;
use lingxi_llm_client::{
    audio::AudioInput,
    openrouter_audio::{
        OpenRouterInputAudioFormat, OpenRouterTranscriptionEncoding,
        OpenRouterTranscriptionRequest,
    },
    RequestOptions,
};

let audio = AudioInput::from_bytes("meeting.wav", "audio/wav", Bytes::from(audio_bytes));
let mut request = OpenRouterTranscriptionRequest::new(
    "openai/whisper-1",
    OpenRouterInputAudioFormat::Wav,
);
request.encoding = OpenRouterTranscriptionEncoding::Base64Json;
request.language = Some("en".into());

let transcript = client
    .openrouter_audio()
    .transcribe(audio, &request, &options)
    .await?;
println!("{}", transcript.text);
```

STT 支持两种明确的请求线格式。`Base64Json` 会把音频编码到 `input_audio.data`，并用 `input_audio.format` 声明 `wav`、`mp3`、`flac`、`m4a`、`ogg`、`webm` 或 `aac`。`Multipart` 会发送 `file` 和 `model` 字段，文件上限为 25 MB。Multipart 会按 `AudioInput.size_bytes` 一次性流式读取 `AudioInput.body`；传输层必须实现 `send_stream`。音频流长度不足、超出声明值或中断时不会重试，提供方可能已收到部分请求，结果会标记为未知。JSON 路径在本 crate 内限制为 50 MB，以限制缓冲和编码开销。两种路径均返回 JSON；`OpenRouterTranscription` 保留 `text`、可选 usage、`X-Generation-Id` 和完整原生 JSON。时间戳仅可与 `verbose_json` 一起请求。当前 multipart 线格式不接受 `provider` passthrough，需改用 base64 JSON。

```rust,ignore
use lingxi_llm_client::openrouter_audio::{
    OpenRouterInputAudioFormat, OpenRouterSpeechFormat, OpenRouterSpeechInputReferences,
    OpenRouterSpeechRequest,
};

let mut request = OpenRouterSpeechRequest::new(
    "mistralai/voxtral-mini-tts-2603",
    "Hello from OpenRouter.",
);
request.voice = Some("en_paul_neutral".into());
request.response_format = OpenRouterSpeechFormat::Mp3;
request.input_references = Some(
    OpenRouterSpeechInputReferences::new(OpenRouterInputAudioFormat::Wav, reference_audio_base64)
        .with_transcript("参考音频对应的文本。"),
);

let output = client
    .openrouter_audio()
    .speak(&request, &options)
    .await?;
save_audio(output.content_type, output.bytes).await?;
```

TTS 成功时返回原始音频字节，不是 JSON。请求格式为 `mp3` 或 `pcm`；模型决定 voice 是否必需及可用值。响应保留服务端 `Content-Type`、`X-Generation-Id` 和调用方提供的账户范围。接口不会自动重试；传输中断可能意味着上游已处理请求，错误的 `dispatch()` 会标记这种不确定结果。

无状态声音克隆可以设置 `input_references`，传入一个强类型 Base64 音频引用和可选转录文本。服务会生成文档规定的 `input_references` 数组，其中包含 `input_audio` data URI，以及可选的后置 `text` 部分。Base64 音频上限为 20 MiB（解码后 15 MiB）；格式错误或超限会在 HTTP 请求前拒绝。音频格式支持和声音克隆权限由模型及 provider 决定。Debug 输出会隐藏音频和转录内容。

OpenRouter 官方说明：[Speech-to-Text](https://openrouter.ai/docs/guides/overview/multimodal/stt)、[Text-to-Speech](https://openrouter.ai/docs/guides/overview/multimodal/tts)。模型发现通过 Models API 的 `output_modalities=transcription` 或 `output_modalities=speech` 筛选；voice、格式和 provider passthrough 支持均取决于具体模型与路由。

合同测试使用本地 mock transport，不会连接 OpenRouter，也不验证真实账户、模型可用性或计费。
