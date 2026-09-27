# OpenRouter Audio Service

`openrouter_audio` exposes OpenRouter's speech-to-text (STT) and text-to-speech (TTS) routes. It calls OpenRouter's `/api/v1/audio/transcriptions` and `/api/v1/audio/speech` endpoints directly. It does not infer audio support from OpenAI services or model catalogs. Each operation receives its OpenRouter bearer key and optional account scope through `RequestOptions`; the scope is returned with the result and is not sent to OpenRouter.

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

STT supports two explicit request formats. `Base64Json` encodes audio as `input_audio.data` and declares its format as `input_audio.format`: `wav`, `mp3`, `flac`, `m4a`, `ogg`, `webm`, or `aac`. `Multipart` sends `file` and `model` fields and is limited to 25 MB. Multipart uploads stream the one-shot `AudioInput.body` using its declared `size_bytes`; use a `Transport` that implements `send_stream`. A short, long, or interrupted source stream is not retried and can leave the provider outcome unknown. This crate caps its buffered JSON path at 50 MB. Both formats return JSON; `OpenRouterTranscription` preserves the transcript, optional usage, `X-Generation-Id`, and full native JSON. Request `verbose_json` to ask for provider-supported timestamps. The current multipart contract does not carry `provider` passthrough; use base64 JSON for those options.

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
        .with_transcript("Transcript of the reference sample."),
);

let output = client
    .openrouter_audio()
    .speak(&request, &options)
    .await?;
save_audio(output.content_type, output.bytes).await?;
```

Successful TTS responses contain raw audio bytes rather than JSON. The typed request supports `mp3` and `pcm`; the selected model determines whether a voice is required and which voice IDs are valid. The result preserves the response `Content-Type`, `X-Generation-Id`, and caller-supplied account scope. The service never retries automatically. If the connection ends after submission, `dispatch()` reports that the request outcome is unknown.

For stateless voice cloning, set `input_references` to one typed Base64 audio reference and an optional transcript. The service emits the documented `input_references` array with an `input_audio` data URI followed by the optional `text` part. Base64 audio is limited to 20 MiB (15 MiB decoded); malformed or oversized values are rejected before HTTP. Audio format support and voice-cloning entitlement remain model/provider-owned. Debug output redacts both the audio and transcript.

Official OpenRouter references: [Speech-to-Text](https://openrouter.ai/docs/guides/overview/multimodal/stt) and [Text-to-Speech](https://openrouter.ai/docs/guides/overview/multimodal/tts). Discover model slugs with the Models API filter `output_modalities=transcription` or `output_modalities=speech`; supported voices, formats, and provider options depend on the selected model and route.

Contract tests use a local mock transport. They do not call OpenRouter or verify real-account access, model availability, or billing.
