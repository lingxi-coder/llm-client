# Vertex AI Gemini speech synthesis

`hosting::vertex::speech` calls Gemini-TTS through the Vertex AI publisher-model REST API. It is separate from Gemini Developer API Interactions TTS and Cloud Text-to-Speech API. The caller explicitly selects the Google Cloud project, location, model, and bearer access token.

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

The service sends one `generateContent` request to `projects/{project}/locations/{location}/publishers/google/models/{model}`. The result exposes decoded, headerless mono PCM16 audio at 24 kHz and preserves Google's native response JSON. It does not play or write audio, fetch credentials, download remote media, or retry an ambiguous request.

The current Vertex guide lists `gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-tts`, `gemini-2.5-flash-lite-preview-tts`, and `gemini-2.5-pro-tts`. The 3.1 preview is listed for `global`; the 2.5 models have a documented regional availability table. The adapter checks the selected model and location against that list. Flash-Lite is single-speaker only; the other listed models support two-speaker dialogue. Google may change model and region availability.

`VertexSpeechRequest::multi_speaker` accepts two speaker/voice declarations and labeled text turns. It renders the dialogue labels into Vertex's `contents` field and uses `generationConfig.speechConfig.multiSpeakerVoiceConfig`. Both request forms accept an optional style prompt. The complete contents field is limited to Google's documented 8,000-byte maximum; the provider may truncate output longer than approximately 655 seconds.

`synthesize_stream` makes one request and receives server-streamed responses through `streamGenerateContent?alt=sse`. Each `next_event()` preserves one native `GenerateContentResponse` and returns any audio bytes decoded from that response. The stream completes successfully only after a candidate supplies a non-empty `finishReason` or Google supplies the documented no-candidate `promptFeedback.blockReason`; EOF before either signal is reported as premature. Typed `finish_reason`/`completion_reason` values and native JSON expose non-`STOP` endings for caller review. This is one-shot server streaming; the service does not send text in multiple input chunks or reconnect a dropped stream.

Google documents Vertex Gemini-TTS output as raw PCM16 at 24 kHz without a WAV header. If a response supplies an audio MIME type, the adapter accepts raw linear PCM labels (`audio/pcm` or `audio/L16`) and rejects conflicting codec, sample-rate, channel, or bit-depth parameters. If MIME metadata is absent, the Vertex TTS contract supplies PCM16/24 kHz/mono metadata. Use another documented API if you need a container format. `account_scope` is a local identity label; Google authentication and token refresh remain with the caller. If `RequestOptions.account_scope` is supplied, it must match the scope.

Sources: [Gemini-TTS on Cloud Text-to-Speech and Vertex AI](https://docs.cloud.google.com/text-to-speech/docs/gemini-tts), [Google's Gemini TTS PCM MIME documentation](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation), [Vertex `streamGenerateContent` REST method](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.publishers.models/streamGenerateContent), [Vertex `GenerateContentResponse` terminal fields](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse), [Google Gen AI SDK streaming route](https://github.com/googleapis/js-genai/blob/main/src/models.ts#L3634-L3717).
