# Qwen 音频生成

`qwen_audio_generation` 为 Alibaba Model Studio 的 `qwen-audio-3.1-tts-next` 提供独立 AudioGen 调用。它与 [`qwen_tts`](qwen-tts.md) 使用不同协议：AudioGen 接收描述性 `text_prompt` 和最多三个参考音频片段，可生成对话、播客、音效及环境音。此操作为非流式，使用官方确认的北京工作空间端点。

服务需要显式工作空间作用域，并在每次调用时提供北京 Model Studio API Key。路由固定为 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer`；客户端不推断其他区域或端点。如果设置了 `RequestOptions::account_scope`，必须与作用域一致，否则会在发送请求前失败。

```rust,no_run
use lingxi_llm_client::{
    HttpTransport,
    protocol::Secret,
    providers::qwen::audio_generation::{
        QwenAudioGenerationFormat, QwenAudioGenerationReference,
        QwenAudioGenerationRequest, QwenAudioGenerationScope,
        QwenAudioGenerationService, QwenAudioReferenceFormat,
    },
    RequestOptions,
};

async fn synthesize(
    api_key: String,
    reference_wav: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let http = HttpTransport::new()?;
    let scope = QwenAudioGenerationScope::new(
        "qwen-beijing",
        "account-production",
        "workspace-abc123",
    )?;
    let service = QwenAudioGenerationService::new(&http, scope)?;
    let reference = QwenAudioGenerationReference::from_bytes(
        QwenAudioReferenceFormat::Wav,
        reference_wav.into(),
    )?;
    let request = QwenAudioGenerationRequest::new(
        "@voice1 说：欢迎。随后响起轻柔的钢琴声。",
    )
    .with_reference(reference)
    .with_format(QwenAudioGenerationFormat::Wav);
    let options = RequestOptions {
        credential: Some(Secret::new(api_key)),
        account_scope: Some("account-production".into()),
        ..Default::default()
    };
    let result = service.synthesize(&request, &options).await?;
    println!("audio URL: {}", result.audio_url());
    Ok(())
}

```

`text_prompt` 必填，最多 3,000 个字符。`@voice1`、`@voice2` 和 `@voice3` 按数组顺序引用参考音频；客户端会在发送前检查被引用的片段是否存在。每个参考项可以是 Model Studio 可读取的 HTTP(S) URL，也可以是内联 WAV、MP3 或 OGG Opus 字节（客户端编码为 data URI）。供应商限制最多三个片段，每段不超过 30 秒、10 MB。客户端对内联音频采用较保守的 10,000,000 字节上限，对参考 URL 设 16 KiB 上限，但不会解析音频时长。URL 参考会原样交给 Model Studio；客户端不会下载。

可选输出控制均为类型化参数：WAV/MP3/PCM、8/16/24/44.1/48 kHz、单声道/立体声、0–100 音量、0.5–2.0 语速、随机种子及 AIGC 标识。`with_constant_bitrate` 和 `with_vbr_quality` 分别选择 MP3 CBR 和 VBR。CBR 码率单位为 kbps，支持值取决于采样率，供应商可能会约束不支持的值。VBR 质量范围为 0–9，0 表示最高质量。未设置的选项会省略，由 Model Studio 使用其文档默认值。

结果保留请求 ID、音频 ID、过期时间、时长、完成原因、用量时长及完整原生响应。签名音频 URL 有效期为 24 小时。服务仅返回 URL，不自动下载、重试或切换区域。POST 开始后发生传输错误时，结果状态为未知，调用方不应自动重发。

此操作返回完整音频。Qwen3-TTS 流式合成和独立实时 TTS 见 [`qwen_tts`](qwen-tts.md) 与 [`qwen_tts_realtime`](qwen-tts-realtime.md)。

官方参考：[Audio Generation API](https://help.aliyun.com/en/model-studio/audio-generation-api)、[qwen-audio-3.1-tts-next 模型信息](https://help.aliyun.com/en/model-studio/qwen-audio-3-1-tts-next)、[Audio generation examples and prompt guide](https://help.aliyun.com/en/model-studio/audio-generation)。
