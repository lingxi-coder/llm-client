# Audio：文件转写、翻译与 TTS

[English](audio.en.md)

先用 `client.provider::<OpenAiClient>(profile)?` 绑定具体 profile，再通过 `provider.audio()` 调用资源。每次操作传入 `RequestOptions`，client 不保存凭证。

`provider.audio()` 是独立于 Chat 的 OpenAI [文件转写](https://developers.openai.com/api/docs/guides/speech-to-text)、[翻译](https://developers.openai.com/api/reference/python/resources/audio/subresources/translations/methods/create)与[文字转语音](https://developers.openai.com/api/docs/guides/text-to-speech)服务。内置 `openai` profile 配置独立路由，配置 v3 可继承或禁用。文件内容通过定长 multipart 流式上传，不会退化为整文件缓冲；输入必须为已完成的 1–25,000,000 字节录音，并带支持的文件扩展名和 MIME 类型。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::audio::{AudioInput, AudioTextResult, AudioError, TranscriptionModel, TranscriptionRequest};

async fn transcribe(client: &LlmClient, options: &RequestOptions)
    -> Result<AudioTextResult, Box<dyn std::error::Error>>
{
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let input = AudioInput::from_bytes("meeting.wav", "audio/wav", b"example audio bytes".to_vec());
    let request = TranscriptionRequest::new(TranscriptionModel::GptTranscribe);
    provider.audio().transcribe(input, &request, options).await.map_err(Into::into)
}
```

对于大文件，直接构造 `AudioInput { filename, media_type, size_bytes, body }`，其中 `body` 为一次性 `BoxStream<Result<Bytes, LlmError>>`。声明字节数必须与流中实际字节数一致；缺少流式请求能力的自定义 `Transport` 会在消费输入前拒绝。请求有总超时，客户端不自动重试或切换连接；上传中断可能已被提供方接受，返回 `OutcomeUnknown`。

`TranscriptionRequest` 支持 `gpt-transcribe`、`gpt-4o-transcribe`、`gpt-4o-mini-transcribe`、`gpt-4o-mini-transcribe-2025-12-15`、`gpt-4o-transcribe-diarize` 和 `whisper-1`。前四者在此切片仅开放 JSON；Whisper 支持文本、SRT、VTT、JSON 和 verbose JSON，词/段时间戳需 verbose JSON；diarize 模型可返回 `diarized_json` 的 `segments[].speaker`。使用 diarize 模型处理超过 30 秒的音频时，调用方须启用 `chunking_auto`；客户端不会猜测流的时长。不支持的模型/格式组合发送前拒绝。`translate()` 只使用官方列出的 `whisper-1`，将语音译为英文，可选文本、字幕或 JSON 输出。`AudioTextResult` 保留完整原生 JSON、文本、单个 `language` 或 `gpt-transcribe` 的 `languages[]`、时长、词/段时间戳与说话人字段（若提供方返回）。

需要将片段关联到已知说话人时，为 diarize 请求附上最多四个 `KnownSpeakerReference`。每个引用包含说话人名、受支持格式的文件名与 MIME 类型、音频字节及调用方提供的时长；客户端要求时长为 2–10 秒，并将其按顺序编码成 `known_speaker_names[]` 和 `known_speaker_references[]` multipart 字段，后者是 `data:{mime};base64,...`。引用只与 `gpt-4o-transcribe-diarize` 和 `diarized_json` 一起接受。调用方负责给出真实片段时长；客户端不会解码音频估算时长。此字段也适用于 `transcribe_stream()`，提供方只在完成 segment 后给出 speaker 标签。详见 OpenAI 的 [speaker diarization guide](https://developers.openai.com/api/docs/guides/speech-to-text#speaker-diarization)。

```rust,no_run
use lingxi_llm_client::providers::openai::audio::{
    AudioTextFormat, KnownSpeakerReference, TranscriptionModel, TranscriptionRequest,
};

fn diarized_request(speaker_wav_bytes: Vec<u8>) -> TranscriptionRequest {
    let speaker = KnownSpeakerReference::new(
        "agent",
        "agent.wav",
        "audio/wav",
        4.2,
        speaker_wav_bytes,
    );
    let mut request = TranscriptionRequest::new(TranscriptionModel::Gpt4oTranscribeDiarize);
    request.format = AudioTextFormat::DiarizedJson;
    request.known_speakers.push(speaker);
    request
}
```

`synthesize()` 接收文字、模型、音色、输出格式和可选语速，返回可逐块读取的 `SpeechStream`，不自动写文件或播放。支持 `gpt-4o-mini-tts`、其已列出的日期版本、`tts-1` 和 `tts-1-hd`；旧 TTS 模型的音色集合更小，也不接受 `instructions`。输入限 1–4096 个字符、语速限 0.25–4.0。输出格式为 MP3、Opus、AAC、FLAC、WAV 或 PCM；PCM 是 24 kHz、16 位有符号小端、单声道原始字节。流断开时错误包含已交付的字节数，客户端不会自动重提。应用向最终用户播放 TTS 时，应按[官方说明](https://developers.openai.com/api/docs/guides/text-to-speech)清楚告知声音由 AI 生成。

选择自定义音色必须使用不透明的 `CustomVoiceRef`，不能直接传裸 ID。`create_voice()` 返回 `CustomVoice` 元数据，通过 `voice.reference().clone()` 传入 `SpeechVoice::Custom`。对于之前已获批的 ID，可调用 `provider.audio().voices().import_approved_voice(id, options)`；这是本地导入，需要显式 profile 和 `RequestOptions.account_scope`，不会查询 ID 是否存在或是否已授权。引用可序列化保存，其中绑定 provider、profile、OpenAI 官方 Speech endpoint 指纹和调用方声明的账户范围；实际 HTTP 请求仍只发送 `"voice": {"id":"voice_123abc"}`。合成会在读取凭证或发送 HTTP 前拒绝无效 ID，以及 provider、profile、Speech endpoint 或账户范围不匹配。此范围检查无法证明项目授权、同意录音或音色仍可用。创建音色与同意录音可通过下文的 `provider.audio().voices()` 生命周期 API 完成。[OpenAI 自定义音色说明](https://developers.openai.com/api/docs/guides/custom-voices)。

```rust,no_run
use lingxi_llm_client::{
    providers::openai::audio::{SpeechVoice, VoiceResourceError},
    LlmClient, RequestOptions,
};

fn import_approved_voice(
    client: &LlmClient,
    options: &RequestOptions,
) -> Result<SpeechVoice, Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let reference = provider.audio()
        .voices()
        .import_approved_voice("voice_123abc", options)?;
    Ok(SpeechVoice::Custom(reference))
}
```

`provider.audio().voices()` 提供官方文档列出的同意短语目录、同意录音创建及其 CRUD，以及自定义音色创建。OpenAI 目前只为音色资源列出创建端点，因此客户端不猜测音色的列举、读取、更新或删除操作。短语目录接口没有公开响应 schema，客户端原样返回 JSON。资源路径从所选 OpenAI audio route 派生，并使用该 profile 的 Bearer 凭证；整个生命周期应使用同一项目级 API key 和 profile，并在每次请求中设置相同且稳定、非敏感的 `RequestOptions.account_scope`。返回的 `VoiceConsentRef` 会将同意 ID 绑定到 profile、endpoint 和调用方声明的账户范围；读取、更新、删除以及创建音色时，若范围不匹配会在发送 HTTP 请求前拒绝。此检查只比较调用方声明的范围，不能证明 API key 实际属于所标识的项目。读取短语需要 `api.voices.read`；创建同意和音色需要 `api.voices.write` 以及自定义音色访问权限。自定义音色仅向符合条件的客户开放。

同意录音和样本录音必须分开录制，并由同一说话人提供。客户端按文档中的 multipart 字段流式上传；每个文件必须非空且不超过 10 MiB，并使用 OpenAI 支持的 audio MIME 类型：`audio/mpeg`、`audio/wav`、`audio/x-wav`、`audio/ogg`、`audio/aac`、`audio/flac`、`audio/webm` 或 `audio/mp4`。OpenAI 要求样本不超过 30 秒，且同意录音必须逐字包含受支持的同意短语；录音内容和访问权限仍由服务端校验。同意资源支持列举、读取、仅更新名称的元数据，以及带删除回执的删除。分页由调用方控制，客户端不会自动请求下一页。

```rust,no_run
use lingxi_llm_client::{
    providers::openai::audio::{AudioInput, CustomVoiceCreateRequest, CustomVoiceRef, VoiceConsentCreateRequest, VoiceResourceError},
    LlmClient, RequestOptions,
};

async fn create_voice(
    client: &LlmClient,
    options: &RequestOptions,
    consent_recording: Vec<u8>,
    sample_recording: Vec<u8>,
) -> Result<CustomVoiceRef, Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    // options.account_scope 使用此 OpenAI 项目稳定、非敏感的标识。
    let voices = provider.audio().voices();
    let _current_phrases = voices.list_consent_phrases(options).await?;
    let consent = voices
        .create_consent(
            &VoiceConsentCreateRequest {
                name: "Speaker consent".into(),
                language: "en-US".into(),
            },
            AudioInput::from_bytes("consent.wav", "audio/wav", consent_recording),
            options,
        )
        .await?;
    let voice = voices
        .create_voice(
            &CustomVoiceCreateRequest {
                name: "Speaker voice".into(),
                consent: consent.reference().clone(),
            },
            AudioInput::from_bytes("sample.wav", "audio/wav", sample_recording),
            options,
        )
        .await?;
    Ok(voice.reference().clone())
}
```

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::audio::{AudioError, SpeechFormat, SpeechModel, SpeechRequest, SpeechVoice};

async fn speak(client: &LlmClient, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let request = SpeechRequest {
        model: SpeechModel::Gpt4oMiniTts,
        input: "你好".into(),
        voice: SpeechVoice::Coral,
        format: SpeechFormat::Pcm,
        instructions: None,
        speed: None,
    };
    let mut audio = provider.audio().synthesize(&request, options).await?;
    while let Some(chunk) = audio.next_chunk().await.map_err(|error| AudioError::Llm(*error.source))? {
        let _bytes = chunk;
    }
    Ok(())
}
```

文件转写也可调用 `provider.audio().transcribe_stream(input, &request, options)`。它发送 `stream=true`，逐个返回原生 `transcript.text.delta`、`transcript.text.segment`（说话人模式）和最终的 `transcript.text.done` 事件；完整文本以 done 事件为准。接口仅接受 GPT 转写模型和 JSON/diarized JSON 输出，拒绝 `whisper-1`。`TranscriptionEventStream::next_event()` 在缺失终态事件、传输断开或无效事件时返回错误；上传为一次性的，客户端不重提或尝试恢复。参考[OpenAI Docs 文件转写流说明](https://developers.openai.com/api/docs/guides/speech-to-text)。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::audio::{AudioInput, TranscriptionModel, TranscriptionRequest};

async fn stream_transcript(client: &LlmClient, options: &RequestOptions)
    -> Result<(), Box<dyn std::error::Error>>
{
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let input = AudioInput::from_bytes("meeting.wav", "audio/wav", b"example audio bytes".to_vec());
    let request = TranscriptionRequest::new(TranscriptionModel::GptTranscribe);
    let mut events = provider.audio().transcribe_stream(input, &request, options).await?;
    while let Some(event) = events.next_event().await? {
        if event.terminal { let _complete_text = &event.native["text"]; }
    }
    Ok(())
}
```

`synthesize_stream()` 可按 SSE 事件读取同一 Speech 请求，只有 GPT TTS 模型可用，并且必须使用官方 OpenAI `/v1/audio/speech` 路由；官方文档不支持 `tts-1` / `tts-1-hd` 的 SSE。它将 `speech.audio.delta.audio` Base64 解码为字节，并将终态 `speech.audio.done.usage` 原样保留。成功 done 必须含整数 `input_tokens`、`output_tokens` 和 `total_tokens`；未识别事件（包括未文档化的流内错误）保留 `event_type`、原生 JSON 和原始 SSE data，不会猜测错误字段。每个 SSE event 最大 8 MiB；连接中断或到达 EOF 前没有有效的 done 时，`next_event()` 返回 `SpeechEventStreamError::Interrupted`，已知事件字段缺失或 Base64 无效时返回 `InvalidEvent`；非 2xx HTTP 状态仍是 `AudioError::Provider`。截至 2026-09-27，本项目复核的 OpenAI Speech API 文档未建立 SSE 文本对齐字段契约，因此本 API 不推测对齐信息；这不等同于断言服务端永远不支持。此 Speech API 不包含异步音频任务、Chat 音频或 Realtime；这些能力由独立服务处理。没有调用真实 OpenAI 账户。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::audio::{SpeechEvent, SpeechModel, SpeechRequest, SpeechVoice};

async fn speak_events(client: &LlmClient, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let request = SpeechRequest {
        model: SpeechModel::Gpt4oMiniTts,
        input: "你好".into(),
        voice: SpeechVoice::Coral,
        format: Default::default(),
        instructions: None,
        speed: None,
    };
    let mut events = provider.audio().synthesize_stream(&request, options).await?;
    while let Some(event) = events.next_event().await? {
        match event {
            SpeechEvent::AudioDelta { audio, .. } => { let _chunk = audio; }
            SpeechEvent::AudioDone { usage, .. } => { let _usage = usage; }
            SpeechEvent::Unknown { event_type, raw_data, .. } => {
                let _unrecognized = (event_type, raw_data);
            }
        }
    }
    Ok(())
}
```
