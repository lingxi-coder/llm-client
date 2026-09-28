# Qwen Audio Generation

`qwen_audio_generation` exposes Alibaba Model Studio's standalone AudioGen operation for `qwen-audio-3.1-tts-next`. It is independent of [`qwen_tts`](qwen-tts.en.md): AudioGen accepts an expressive `text_prompt` and up to three reference clips, and can create dialogue, podcasts, sound effects, and ambience. The operation is non-streaming and uses the Beijing workspace endpoint documented for this API.

The service requires an explicit workspace scope and a Beijing Model Studio API key on every call. Its route is fixed to `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer`; the client does not infer any other region or endpoint. If `RequestOptions::account_scope` is supplied, it must match the scope before the request is sent.

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
        "@voice1 says: Welcome. A soft piano note follows.",
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

`text_prompt` is required and limited to 3,000 characters. The `@voice1`, `@voice2`, and `@voice3` markers refer to clips in reference-array order; the client checks that each referenced clip exists before sending. Each reference is either an HTTP(S) URL readable by Model Studio or inline WAV, MP3, or OGG Opus bytes encoded as a data URI. The provider accepts up to three clips, each no longer than 30 seconds and no larger than 10 MB. This client applies a conservative 10,000,000-byte cap to inline clips and a 16 KiB cap to URLs, but it does not inspect clip duration. URL references are passed through; the client does not fetch them.

Typed optional output controls cover WAV/MP3/PCM, 8/16/24/44.1/48 kHz, mono/stereo, volume 0–100, speech rate 0.5–2.0, a seed, and the AIGC tag. `with_constant_bitrate` and `with_vbr_quality` select MP3 CBR and VBR. CBR bitrate is in kbps; supported levels depend on sample rate, and the provider may constrain unsupported values. VBR quality is 0–9, with 0 representing the highest quality. Omitted options leave the documented service defaults in effect.

The result preserves request and audio IDs, expiration, duration, finish reason, usage duration, and the native response. Its signed audio URL is valid for 24 hours. The service returns the URL without downloading it, retrying, or switching regions. A transport failure after POST begins is an unknown outcome and must not be automatically retried.

This operation returns complete audio. For Qwen3-TTS streaming and independent real-time TTS, see [`qwen_tts`](qwen-tts.en.md) and [`qwen_tts_realtime`](qwen-tts-realtime.en.md).

Official references: [Audio Generation API](https://help.aliyun.com/en/model-studio/audio-generation-api), [qwen-audio-3.1-tts-next model information](https://help.aliyun.com/en/model-studio/qwen-audio-3-1-tts-next), and [Audio generation examples and prompt guide](https://help.aliyun.com/en/model-studio/audio-generation).
