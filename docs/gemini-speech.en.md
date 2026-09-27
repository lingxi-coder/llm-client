# Gemini native speech synthesis

`gemini_speech` wraps unary and SSE text-to-speech through Google's Gemini Developer API. Google's current TTS guide uses the Interactions API:

```text
POST https://generativelanguage.googleapis.com/v1beta/interactions
```

This is neither Generate Content's audio-generation path nor the Live API for real-time conversation. This adapter covers **single-speaker and multi-speaker unary and streaming** synthesis with Gemini 3.8 Flash TTS and Gemini 3.8 Flash-Lite TTS.

The service binds a profile, non-secret Google Cloud project/account scope, and Google's official Interactions endpoint. Pass `&Secret<String>` to every unary or streaming synthesis call; the service does not store an API key. A transport failure may occur after Google receives the request, so the outcome is marked unknown and the service never retries.

## Request and result

Requests follow the TTS guide's Interactions JSON: `model`, `input` text with an optional `speech_metadata.style` annotation, `response_format`, and `generation_config.speech_config`. Unary requests set `stream: false`; streaming requests set `stream: true`. Callers must choose a published model, voice, format, and sample rate; enums prevent sending unsupported model and format values.

Supported output MIME types are `audio/wav`, `audio/l16`, `audio/mulaw`, and `audio/alaw`. The adapter accepts 8 kHz, 16 kHz, and 24 kHz sample rates shown in Google's guide. Default unary `audio/wav` is a WAV file with a RIFF header; streaming `audio/l16` is headerless 16-bit little-endian PCM.

Google returns base64 audio in the completed Interaction's `steps[].content[]`. The service decodes the documented audio field into `audio` bytes and retains the complete JSON response, usage, Interaction ID, and provider audio block in `native`. Missing audio or invalid base64 is reported as an invalid response after acceptance.

```rust,ignore
use lingxi_llm_client::{
    gemini_speech::{
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

For a two-speaker dialogue, use `GeminiSpeechSpeaker` to map two speaker labels to prebuilt voices and `GeminiSpeechTurn` for each separately labeled text segment. Each turn may carry its own style. The request sends `speech_config.mode: "conversational"`; every turn must name one configured speaker. This API follows Google's single-request multi-speaker limit of two speakers and requires prebuilt voices. Designed or replicated custom voices should be synthesized in separate per-speaker calls.

```rust,ignore
use lingxi_llm_client::gemini_speech::{
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

## Streaming synthesis

Interactions streaming uses SSE. For TTS, an audio `step.delta` has base64 audio in `delta.data`; the service decodes each increment to `audio_chunk` and retains the complete provider event in `native`. Gemini 3.8 TTS defaults to headerless `audio/l16` at 24 kHz mono for streaming. Use `GeminiSpeechRequest::new_streaming` or `new_multi_speaker_streaming` to make that default explicit, or select another documented format and sample rate.

```rust,ignore
use lingxi_llm_client::{
    gemini_speech::{GeminiSpeechModel, GeminiSpeechRequest, GeminiSpeechScope, GeminiSpeechService, GEMINI_SPEECH_ENDPOINT},
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
    // event.native preserves the original event JSON; event.event_type identifies its lifecycle step.
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The service does not concatenate a complete audio artifact in memory. It limits each SSE event to 8 MiB, one response to 96 MiB of wire data, and decoded audio to 64 MiB; reads also remain subject to the 300-second request deadline. It requires the documented `done` / `[DONE]` marker after `interaction.completed`. Missing markers, transport interruption, provider `error` events, malformed data, and limit violations are reported explicitly. An interruption error carries the Interaction ID, a profile/account/endpoint-scoped reference, the latest `event_id` cursor, and the number of audio bytes already yielded. The service does not reconnect or retry. If Google still exposes the Interaction, callers may decide whether to resume with the official GET `stream=true&last_event_id=...` contract.

## Current boundary

The service does not manage Voice Design or Voice Replication resources. For a custom voice, create and manage it through Google's Voices API, then pass its returned voice ID to a single-speaker request. Google's Live API handles bidirectional real-time voice interaction and belongs in a separate Live service.

The request and response have local size bounds to prevent unbounded memory use. Google remains authoritative for model input-token limits. Mock-transport tests make no live API calls.

## Official references

- [Gemini Text-to-speech generation](https://ai.google.dev/gemini-api/docs/speech-generation): TTS models, Interactions request, voices, audio formats, and limits.
- [Streaming interactions](https://ai.google.dev/gemini-api/docs/streaming): SSE sequence, `interaction.completed`, `error`, and `[DONE]` events.
- [Gemini Interactions API](https://ai.google.dev/api/interactions-api): REST resource and native response steps.
- [Gemini API release notes](https://ai.google.dev/gemini-api/docs/changelog): GA model IDs for Gemini 3.8 TTS.
- [Gemini Voices API](https://ai.google.dev/api/voices): prebuilt voice library and custom voice resource management.
