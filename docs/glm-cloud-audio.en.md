# GLM Hosted Cloud Audio

`glm_cloud_audio` wraps Zhipu's hosted BigModel audio routes. It is separate from [`glm_audio`](glm-audio.md), which calls a caller-owned SGLang server and does not use the hosted endpoints.

| Operation | Mainland China | International |
| --- | --- | --- |
| ASR | `POST https://open.bigmodel.cn/api/paas/v4/audio/transcriptions` | `POST https://api.z.ai/api/paas/v4/audio/transcriptions` |
| TTS | `POST https://open.bigmodel.cn/api/paas/v4/audio/speech` | Not established; rejected before sending because no international TTS route/schema is published |

The service is constructed for one region and non-secret account scope. Pass `&Secret<String>` to each `transcribe` or `synthesize` call; the service does not retain credentials. It never retries. A transport failure after submission is reported as an unknown outcome because the provider may have received and billed the request.

## ASR

ASR uses `glm-asr-2512` and multipart form data containing `file`, `model`, and `stream`, with optional `request_id` and `user_id`. The international API documents WAV and MP3 input, a 25 MB maximum file size, and a 30-second duration ceiling. This client checks the file size and filename extension; it does not decode media to inspect duration, so the provider remains authoritative for that limit.

When `stream` is false, the result is normalized from the regional documented response shape while retaining the native JSON. Mainland BigModel returns the SDK's completion shape; the international endpoint returns a `text` field. When `stream` is true, `GlmCloudTranscriptionOutput::Streaming` exposes the response content type and raw bytes. Callers own event framing and parsing; the adapter does not assume an undocumented chunk encoding.

```rust,ignore
use bytes::Bytes;
use lingxi_llm_client::{
    audio::AudioInput,
    glm_cloud_audio::{
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

Hosted TTS uses the mainland `/audio/speech` route and the `glm-tts` model. The wrapper sends the first-party SDK's JSON field names: `model`, `input`, `encode_format`, `stream`, and optional `voice`, `response_format`, `request_id`, and `user_id`. It does not add undocumented controls such as speed or volume. The adapter caps text at 100,000 Unicode characters to bound local request memory; this is not a provider model limit.

International TTS is **not established**, rather than documented as unsupported. The current Z.AI [API index](https://docs.z.ai/llms.txt) and [OpenAPI](https://docs.z.ai/openapi.json) publish `/paas/v4/audio/transcriptions` but no speech route or TTS request schema. The official [Java SDK README](https://github.com/zai-org/z-ai-sdk-java) describes an Audio service with TTS and STT, and its [0.3.5 release notes](https://github.com/zai-org/z-ai-sdk-java/releases) mention a TTS/ASR API refactor and samples. Those summaries do not identify an international speech endpoint or its request contract, so the client keeps the preflight rejection until the exact Z.AI wire is published. The published OpenAPI bearer security applies to its documented operations; it does not establish authentication or model entitlement for an unpublished TTS route.

The response is returned as `GlmCloudAudioByteStream`. This type preserves each HTTP body chunk as bytes, including NUL and non-UTF-8 values; it does not buffer the full audio or parse it as JSON. For `stream: true`, inspect `content_type()` and handle the provider's stream format in the caller. Dropping the body stream cancels the remaining response read.

```rust,ignore
use futures::StreamExt;
use lingxi_llm_client::glm_cloud_audio::GlmCloudSpeechRequest;

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

## Documentation basis

- [BigModel API introduction](https://docs.bigmodel.cn/cn/api/introduction) publishes the mainland API root and bearer authentication.
- [BigModel speech-to-text reference](https://docs.bigmodel.cn/api-reference/模型-api/语音转文本) and [text-to-speech reference](https://docs.bigmodel.cn/api-reference/模型-api/文本转语音) list the hosted ASR and TTS operations. The ASR guide names GLM-ASR-2512; the TTS guide names GLM-TTS.
- The [official Zhipu Python SDK audio implementation](https://github.com/MetaGLM/zhipuai-sdk-python-v4/blob/main/zhipuai/api_resource/audio/audio.py) corroborates the `/audio/speech` route, JSON field names, binary response, and streaming response support. Its [transcription implementation](https://github.com/MetaGLM/zhipuai-sdk-python-v4/blob/main/zhipuai/api_resource/audio/transcriptions.py) corroborates the multipart route and fields used here.
- The [official Java SDK `ModelTTS` constant](https://github.com/zai-org/z-ai-sdk-java/blob/main/core/src/main/java/ai/z/openapi/core/Constants.java) identifies `glm-tts`; the SDK README lists both API roots. These identify a model and endpoints, not an international TTS route/schema.
- [Z.AI's transcription API reference](https://docs.z.ai/api-reference/audio/audio-transcriptions) and [GLM-ASR-2512 guide](https://docs.z.ai/guides/audio/glm-asr-2512) publish the international route, model ID, upload formats and limits, and streaming option.
- The current [Z.AI OpenAPI](https://docs.z.ai/openapi.json) declares the international production API server and bearer security, and its audio paths include transcription only. The [API index](https://docs.z.ai/llms.txt) likewise lists Audio Transcriptions only. The [official Java SDK README](https://github.com/zai-org/z-ai-sdk-java) and [release history](https://github.com/zai-org/z-ai-sdk-java/releases) signal that SDK audio includes TTS, but do not establish the international endpoint or exact wire schema. The OpenAPI shape supports a precise “not established” conclusion, not a claim of vendor non-support.

Tests use mocked transports only. They verify route selection, request encoding, preflight rejection, unknown-write semantics, and byte preservation; they do not establish live account access or provider availability.
