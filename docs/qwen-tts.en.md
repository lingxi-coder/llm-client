# Qwen HTTP speech synthesis and SSE audio output

`QwenTtsService` submits a complete text input to Model Studio's `qwen3-tts-flash`. `synthesize()` returns the temporary audio URL; `synthesize_stream()` returns native SSE audio deltas and the final URL. Neither mode downloads audio, retries, or switches accounts automatically. HTTP SSE streams output only; standalone bidirectional Qwen TTS Realtime uses a different protocol.

```rust,no_run
use lingxi_llm_client::{
    HttpTransport, RequestOptions,
    protocol::Secret,
    providers::qwen::tts::{
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
    let request = QwenTtsRequest::new("Hello, welcome to speech synthesis.", "Cherry")
        .with_language_type(QwenTtsLanguage::English);

    let result = service.synthesize(&request, &options).await?;
    println!("audio ID: {:?}", result.audio_id);
    println!("temporary audio URL: {}", result.audio_url());
    Ok(())
}
```

`QwenTtsScope` binds a profile, account, region, and workspace identity. Pass a key for that selected region on each call. If `RequestOptions::account_scope` is set, it must match the scope. The regional DashScope hosts are Beijing `https://dashscope.aliyuncs.com` and Singapore `https://dashscope-intl.aliyuncs.com`; this module appends Qwen-TTS's `/api/v1/services/aigc/multimodal-generation/generation` path. The Qwen-TTS API page contains a Singapore-labeled example that still uses the Beijing host. This client follows Model Studio's regional endpoint and API-key tables so a Singapore key is sent to the Singapore host. The Qwen-TTS synthesis guide does not define a workspace-specific route for this model family, so the workspace ID is retained as caller-owned scope metadata. Qwen-Audio-TTS's `/SpeechSynthesizer` interface is a separate model and wire contract.

`QwenTtsRequest` sends the full text, required voice name, and optional `language_type`. Documented language values are `Auto`, Chinese, English, German, Italian, Portuguese, Spanish, Japanese, Korean, French, and Russian. This slice accepts at most 600 Unicode characters, matching the current Qwen3-TTS-Flash API limit. Other model families and instruction-control fields are not silently added to this request.

The buffered result and the SSE completion both use `QwenTtsSynthesis`, which preserves the request ID, audio ID, `finish_reason`, character usage, expiration timestamp, and the exact provider audio URL. Non-streaming output provides the complete audio URL for 24 hours; `audio.data` is documented as empty in this mode. `Debug` redacts spoken text and the signed URL. Use the URL while it is valid to download audio through a caller-managed HTTP client; this service does not follow it.

Each call sends at most one synthesis request. `QwenTtsError::dispatch_outcome()` distinguishes local `NotSent` failures, explicit HTTP 4xx except 408 `Rejected` responses, ambiguous transport/408/5xx `Unknown` outcomes, and malformed successful `Accepted` responses. Errors after a successful SSE handshake are `Accepted`; this includes an interrupted body or a native provider error. `StreamProvider` retains the full provider error JSON, including usage and extensions, while redacting it from `Debug`. HTTP error diagnostics retain `x-request-id` when the body omits the request ID. `Unknown` or `Accepted` may be billable, so do not blindly replay them. Contract tests use a mock transport; no live account was called.

## Stream audio from a complete text

`synthesize_stream()` uses the same model, full text, voice, language, region and account checks as `synthesize()`. It adds `X-DashScope-SSE: enable` and requests `text/event-stream`; it does not add an OpenAI-style `stream` body field.

```rust,no_run
use lingxi_llm_client::{
    RequestOptions,
    providers::qwen::tts::{QwenTtsRequest, QwenTtsService, QwenTtsStreamEventKind},
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

Every event's `native` field retains the exact JSON payload, including request/audio IDs, usage, and provider extensions. Audio segments decode `output.audio.data` from Base64. No WAV header is inserted and no audio is accumulated by the client.

The official completion signal is `output.finish_reason: "stop"`; `null` means generation is still in progress. The documented final payload has empty `output.audio.data` and the full audio URL. A URL alone does not establish completion. `Completed` is emitted only after a valid stop payload and clean HTTP EOF, so late native errors, extra data, malformed payloads and transport failures cannot be hidden behind an early success. Early EOF without stop is an interruption. After an error, the stream releases its body and later reads return `None`. An unterminated final SSE event is decoded at clean EOF.

SSE events are capped at 8 MiB and the total wire response at 64 MiB, with parsing in 64 KiB slices to bound queued events. These are client resource limits, not advertised provider limits. `RequestOptions::total_timeout`, when set, also covers waiting for body chunks and final EOF. Dropping the stream cancels the read without retrying or claiming that the provider stopped billing. No audio device, text-input WebSocket, Qwen-Audio-TTS or CosyVoice support is implied by this HTTP output mode.

Official references: [Qwen-TTS API reference](https://help.aliyun.com/en/model-studio/qwen-tts-api), [non-real-time speech synthesis guide](https://help.aliyun.com/en/model-studio/non-realtime-tts-user-guide), [Model Studio regions and endpoints](https://help.aliyun.com/en/model-studio/regions/), [Base URL by region](https://help.aliyun.com/en/model-studio/base-url), and [Qwen3-TTS-Flash model information](https://help.aliyun.com/en/model-studio/qwen3-tts-flash).
