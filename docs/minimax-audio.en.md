# MiniMax speech transcription

[简体中文](minimax-audio.md)

`client.provider::<MiniMaxClient>(profile)?.audio(endpoint).transcribe(...)` is an independent synchronous file transcription service for MiniMax ASR 1.0. It does not use Chat's `base_url` or OpenAI's transcription fields. The caller must select a full regional endpoint explicitly and provide the matching account API key in `RequestOptions::credential` on every call:

- International: `https://api.minimax.io/v1/speech_to_text`
- Mainland China: `https://api.minimax.cn/v1/speech_to_text`

The service accepts only the `/v1/speech_to_text` path on those two HTTPS hosts. It rejects URL userinfo, queries, fragments, and non-443 ports. The caller chooses the endpoint; the service does not infer it from a chat route, switch regions, or fail over. ASR is stateless, so the service does not retain account scope, credentials, or cached data.

When bound through `MiniMaxClient`, the request uses the authenticator registered for that profile, which must have a non-`none` authentication strategy. Direct construction with `MiniMaxAudioService::new` still sends the supplied API key as a Bearer header.

```rust,no_run
use lingxi_llm_client::{
    providers::openai::audio::AudioInput,
    providers::minimax::audio::{
        MiniMaxAudioService, MiniMaxSpeechLanguage, MiniMaxTranscriptionRequest,
        MINIMAX_ASR_INTERNATIONAL_ENDPOINT,
    },
    protocol::Secret,
    HttpTransport, RequestOptions,
};

async fn transcribe() -> Result<(), Box<dyn std::error::Error>> {
    let transport = HttpTransport::new()?;
    let service = MiniMaxAudioService::new(&transport, MINIMAX_ASR_INTERNATIONAL_ENDPOINT)?;
    let input = AudioInput::from_bytes("meeting.wav", "audio/wav", b"audio bytes".to_vec());
    let request = MiniMaxTranscriptionRequest {
        language: Some(MiniMaxSpeechLanguage::English),
        ..Default::default()
    };
    let options = RequestOptions {
        credential: Some(Secret::new("account API key".into())),
        ..Default::default()
    };
    let result = service.transcribe(input, &request, &options).await?;
    println!("{}", result.text);
    Ok(())
}
```

An existing `LlmClient` or `ClientSnapshot` can also create this service with `client.provider::<MiniMaxClient>(profile)?.audio(endpoint)`. Both constructors require the endpoint explicitly. Do not derive the audio endpoint from a MiniMax Chat `base_url`.

`AudioInput` is a one-shot byte stream with a declared size. MiniMax accepts WAV, AIFF, FLAC, M4A/ALAC, MP3, AAC, Opus, and Ogg containers up to 50 MB and 500 seconds; raw PCM without a container is unsupported. The client checks size, MIME type, filename extension, and actual byte count while sending multipart form data. It cannot infer recording duration from a generic stream, so MiniMax may reject recordings over 500 seconds.

`MiniMaxTranscriptionRequest` selects `json` (default), `verbose_json`, SRT, or VTT output, sentence or word timestamps, and an optional typed language hint. Omitting the hint enables mixed-language recognition. JSON responses preserve the native provider JSON, text, duration, speaker count, diarized segments, and trace ID when present. SRT/VTT responses preserve the full subtitle text. `verbose_json`, SRT, and VTT use the provider's diarization and forced alignment.

`transcribe()` sends `stream=false` and collects one response. For incremental JSON transcription, use `transcribe_stream()`:

```rust,no_run
use futures::StreamExt;

async fn transcribe_incrementally(
    service: &lingxi_llm_client::providers::minimax::audio::MiniMaxAudioService<'_>,
    input: lingxi_llm_client::providers::openai::audio::AudioInput,
    options: &lingxi_llm_client::RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut events = service
        .transcribe_stream(
            input,
            &lingxi_llm_client::providers::minimax::audio::MiniMaxTranscriptionRequest::default(),
            options,
        )
        .await?;
    let mut transcript = String::new();
    let mut duration = None;
    while let Some(event) = events.next().await {
        let event = event?;
        transcript.push_str(&event.delta);
        if event.finish {
            duration = event.duration_seconds;
        }
    }
    println!("{transcript} ({duration:?} seconds)");
    Ok(())
}
```

`transcribe_stream()` sets `stream=true`, requests `text/event-stream`, and accepts only `response_format=json`. It parses arbitrary transport chunk boundaries, requires event indexes to begin at zero and advance by one, and requires the terminal `finish=true` event to carry a finite non-negative duration. The caller concatenates deltas in order; the service stops after the terminal event. `verbose_json`, SRT, and VTT cannot be combined with streaming. The default request deadline is 10 minutes and can be changed with `RequestOptions::total_timeout`.

Uploads are consumed once; the client never retries or fails over automatically. `MiniMaxAudioError::dispatch()` distinguishes local preflight failures (`NotSent`), explicit HTTP 4xx rejection (`Rejected`), uncertain transport or server outcomes (`Unknown`), and malformed responses after a successful HTTP status (`Accepted`). Treat `Unknown` and `Accepted` as potentially billable and do not blindly replay them. No live MiniMax account was called; the service contract is covered with local mock transport tests.

Official references: [Speech to Text — International](https://platform.minimax.io/docs/api-reference/speech-to-text) and [Speech to Text — Mainland China](https://platform.minimax.cn/docs/api-reference/speech-to-text).
