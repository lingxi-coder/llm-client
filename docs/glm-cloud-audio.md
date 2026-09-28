# GLM 托管云端音频

[English](glm-cloud-audio.en.md)

`glm_cloud_audio` 封装智谱 BigModel / Z.AI 托管音频接口。它与 [`glm_audio`](glm-audio.md) 分开：后者调用用户自行部署的 SGLang 服务，不使用智谱托管云端路由。

| 操作 | 中国大陆 | 国际区 |
| --- | --- | --- |
| ASR | `POST https://open.bigmodel.cn/api/paas/v4/audio/transcriptions` | `POST https://api.z.ai/api/paas/v4/audio/transcriptions` |
| TTS | `POST https://open.bigmodel.cn/api/paas/v4/audio/speech` | 尚未建立国际线路契约；因无已公布的国际 TTS 路由/Schema 而在发送前拒绝 |

服务实例绑定一个区域和非秘密账户作用域。每次调用 `transcribe` 或 `synthesize` 时传入 `&Secret<String>`，因此实例不会保存过期凭据。服务不会自动重试。提交后发生传输错误时，结果标记为未知，因为请求可能已到达提供方并产生计费。

## ASR

ASR 使用 `glm-asr-2512`，以 multipart/form-data 发送 `file`、`model` 和 `stream`，并可附带 `request_id`、`user_id`。国际区 API 文档规定音频格式为 WAV 或 MP3，文件最大 25 MB、时长不超过 30 秒。客户端会检查文件大小和扩展名，但不会解码音频检查时长，因此时长限制仍由提供方最终执行。

`stream` 为 `false` 时，适配器按对应区域公开的响应结构提取文本，同时保留原始 JSON。中国大陆 BigModel 返回官方 SDK 使用的 completion 结构；国际区接口返回 `text` 字段。`stream` 为 `true` 时，`GlmCloudTranscriptionOutput::Streaming` 提供响应媒体类型和原始字节，由调用方按响应格式解析事件；适配器不猜测未确认的分块编码。

```rust,ignore
use bytes::Bytes;
use lingxi_llm_client::{
    providers::openai::audio::AudioInput,
    providers::zhipu::cloud_audio::{
        GlmCloudAudioRegion, GlmCloudAudioScope, GlmCloudAudioService,
        GlmCloudTranscriptionOutput, GlmCloudTranscriptionRequest,
    },
    protocol::Secret,
    transport::HttpTransport,
};

let http = HttpTransport::new()?;
let scope = GlmCloudAudioScope::new(
    "zhipu-mainland",
    "account-production",
    GlmCloudAudioRegion::MainlandChina,
)?;
let audio = GlmCloudAudioService::new(&http, scope)?;
let input = AudioInput::from_bytes("meeting.wav", "audio/wav", Bytes::from(audio_bytes));
let request = GlmCloudTranscriptionRequest::new();
let key = Secret::new(api_key);

if let GlmCloudTranscriptionOutput::Complete(transcript) =
    audio.transcribe(&key, input, &request).await?
{
    println!("{}", transcript.text);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## TTS

托管 TTS 使用中国大陆 `/audio/speech` 路由和 `glm-tts` 模型。请求遵循智谱官方 SDK 的 JSON 字段名：`model`、`input`、`encode_format`、`stream`，以及可选的 `voice`、`response_format`、`request_id`、`user_id`。当前适配器不添加官方 SDK 契约中未确认的语速或音量字段。为限制单次请求的本地内存占用，文本输入上限为 100,000 个 Unicode 字符；这不是提供方的模型限制。

国际 TTS 当前属于**契约尚未建立**，不是已有文档明确声明不支持。Z.AI 当前[官方 API 索引](https://docs.z.ai/llms.txt)和[OpenAPI](https://docs.z.ai/openapi.json)公布了 `/paas/v4/audio/transcriptions`，但没有 speech 路由或 TTS 请求 Schema。官方 [Java SDK README](https://github.com/zai-org/z-ai-sdk-java)将 Audio 服务概述为 TTS 与 STT，[0.3.5 发布说明](https://github.com/zai-org/z-ai-sdk-java/releases)也提到 TTS/ASR API 重构和示例；这些摘要没有给出国际 speech endpoint 或其请求契约。因此在厂商公布具体 Z.AI wire 前，客户端继续在发送前拒绝。当前 OpenAPI 的 Bearer 安全声明适用于已文档化的接口，不能用来确定未公布 TTS 路由的认证或模型权限。

返回值 `GlmCloudAudioByteStream` 会逐块保留 HTTP 响应字节，包括 NUL 和非 UTF-8 内容；它不会缓存完整音频，也不会把响应当作 JSON 解析。`stream: true` 时，可检查 `content_type()` 并在调用方解析提供方的流格式。丢弃响应流会取消剩余读取。

```rust,ignore
use futures::StreamExt;
use lingxi_llm_client::providers::zhipu::cloud_audio::GlmCloudSpeechRequest;

let mut request = GlmCloudSpeechRequest::new("你好，欢迎使用语音合成。", "base64");
request.voice = Some("tongtong".into());
request.response_format = Some("wav".into());
let output = audio.synthesize(&key, &request).await?;
let mut chunks = output.into_body();
while let Some(chunk) = chunks.next().await {
    audio_sink.write_all(&chunk?).await?;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## 文档依据

- [BigModel API 接入说明](https://docs.bigmodel.cn/cn/api/introduction)公布中国大陆 API 根地址和 Bearer 鉴权方式。
- [BigModel 语音转文本](https://docs.bigmodel.cn/api-reference/模型-api/语音转文本)与[文本转语音](https://docs.bigmodel.cn/api-reference/模型-api/文本转语音)文档列出托管 ASR、TTS 能力；ASR 指南名称为 GLM-ASR-2512，TTS 指南名称为 GLM-TTS。
- [智谱官方 Python SDK 音频实现](https://github.com/MetaGLM/zhipuai-sdk-python-v4/blob/main/zhipuai/api_resource/audio/audio.py)确认 `/audio/speech` 路径、JSON 字段名、二进制响应和流式响应支持；[转录实现](https://github.com/MetaGLM/zhipuai-sdk-python-v4/blob/main/zhipuai/api_resource/audio/transcriptions.py)确认 multipart 路径和本适配器采用的字段。
- [官方 Java SDK 的 `ModelTTS` 常量](https://github.com/zai-org/z-ai-sdk-java/blob/main/core/src/main/java/ai/z/openapi/core/Constants.java)确认模型 ID 为 `glm-tts`，SDK README 列出了两个 API 根地址。这只能确认模型和 endpoint 根地址，不能确定国际 TTS 路由或 Schema。
- [Z.AI 语音转文本 API](https://docs.z.ai/api-reference/audio/audio-transcriptions)与 [GLM-ASR-2512 指南](https://docs.z.ai/guides/audio/glm-asr-2512)公布国际区路由、模型 ID、上传格式与限制，以及流式参数。
- 当前 [Z.AI OpenAPI](https://docs.z.ai/openapi.json)声明国际生产 API server 和 Bearer 安全；其中音频接口只有转写。[API 索引](https://docs.z.ai/llms.txt)同样只列 Audio Transcriptions。官方 [Java SDK README](https://github.com/zai-org/z-ai-sdk-java)和[发布历史](https://github.com/zai-org/z-ai-sdk-java/releases)表明 SDK 音频能力包含 TTS，但没有确认国际 endpoint 或精确 wire Schema。依据 OpenAPI 只能得出“契约尚未建立”，不能说厂商明确不支持。

测试仅使用模拟传输，检查路由、请求编码、本地预检、未知写入语义和字节保真；它们不代表真实账户权限或线上可用性。
