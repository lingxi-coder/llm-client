# Gemini Voices API

`providers::google::speech::GeminiVoicesService` wraps the Gemini Developer API Voices resource at `POST/GET https://generativelanguage.googleapis.com/v1beta/voices` and `GET/DELETE /v1beta/voices/{voice_id}`. It supports prompted custom voices, replicated voices from reference and consent recordings, catalog listing and filtering, pagination, stored-voice lookup, and stored-voice deletion. This resource is separate from TTS Interactions and the Live API.

Each service is scoped to a Google profile, caller-supplied account/project identity, and the exact Voices endpoint. Pass `&Secret<String>` to each call; the service does not retain API keys. Stored-voice references are bound to that scope. `GeminiVoice::speech_request` transfers a voice ID or stateless key into an Interactions TTS request only when the Google provider, profile, and account match the supplied `GeminiSpeechScope`.

```rust,ignore
use bytes::Bytes;
use lingxi_llm_client::{
    providers::google::speech::{
        GeminiSpeechModel, GeminiSpeechScope, GeminiVoicesScope, GeminiVoicesService,
        GeminiVoiceAudioData, GeminiVoiceCreateRequest, GeminiVoiceListOptions,
        GEMINI_SPEECH_ENDPOINT, GEMINI_VOICES_ENDPOINT,
    },
    protocol::Secret,
    transport::HttpTransport,
};

let http = HttpTransport::new()?;
let scope = GeminiVoicesScope::new(
    "gemini-tts",
    "google-project-prod",
    GEMINI_VOICES_ENDPOINT,
)?;
let voices = GeminiVoicesService::new(&http, scope)?;
let key = Secret::new(api_key);

let created = voices
    .create(&key, &GeminiVoiceCreateRequest::prompted(
        "A calm, warm narrator with a low register.",
    ).with_display_name("Warm narrator"))
    .await?;

let page = voices
    .list(&key, &GeminiVoiceListOptions::new()
        .with_page_size(100)
        .with_types([lingxi_llm_client::providers::google::speech::GeminiVoiceType::Prompted]))
    .await?;
let _next_page_token = page.next_page_token;

let speech_scope = GeminiSpeechScope::new(
    "gemini-tts",
    "google-project-prod",
    GEMINI_SPEECH_ENDPOINT,
)?;
let request = created.speech_request(
    &speech_scope,
    GeminiSpeechModel::Gemini38FlashTts,
    "Hello from this custom voice.",
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For replication, the host supplies both recordings. This example asks Google for a stateless key (`store = false`); the key remains bound to the source profile/account when converted into a speech request.

```rust,ignore
let source = GeminiVoiceAudioData::new(Bytes::from(reference_wav), "audio/wav");
let consent = GeminiVoiceAudioData::new(Bytes::from(consent_wav), "audio/wav");
let replicated = voices
    .create(&key, &GeminiVoiceCreateRequest::replicated(source, consent, false))
    .await?;
let request = replicated.speech_request(
    &speech_scope,
    GeminiSpeechModel::Gemini38FlashTts,
    "Hello from this replicated voice.",
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Prompted creation always sends `store: true`, because Google rejects prompted voices that are not stored. For replication, provide both the speaker reference recording and the consent recording to `GeminiVoiceCreateRequest::replicated(source, consent, store)`. Both are required and use raw bytes plus an IANA MIME type; the service base64-encodes them in one JSON request and does not retain the recordings. Google's official guide requires a real adult speaker, recommends a clean 10–30 second reference clip, and requires the same speaker to recite the exact consent statement in one of the supported languages. The host application is responsible for obtaining and verifying consent before calling `create`.

With `store = true`, Google returns a managed `voice_...` ID that can be listed, fetched, and deleted. With `store = false`, replicated voices return a caller-managed `voicekey_...`; it is not a managed resource and cannot be fetched or deleted through the Voices API. Google currently documents a quota of 200 stored custom voices per project with a one-year lifetime, and a seven-day lifetime for stateless keys. The Voices API reference and guide disagree about the default for `store`, so this adapter always sends the choice explicitly.

`GeminiVoiceListOptions` exposes the documented repeated filters (`accent`, `context`, `gender`, `language_code`, `persona`, `pitch`, `region_code`, and `type`), `search`, `page_size`, and `page_token`. Page size is locally capped at the documented maximum of 1000; search is limited to the documented 2048-byte maximum. Keep filters unchanged between pages and pass `next_page_token` through unchanged. Prebuilt voices appear in list results, but Google does not support fetching or deleting them individually.

The client does not retry, poll, or download implicitly. Transport failures, HTTP 408/5xx, and malformed successful create responses are surfaced as unknown/accepted outcomes, so callers can reconcile with a stored-voice list before deciding whether to retry. Request JSON is capped locally at 32 MiB and responses at 64 MiB to bound in-memory encoding and decoding; these are client safety limits, not Google API limits.

## Official references

- [Gemini Voices API reference](https://ai.google.dev/api/voices)
- [Voice replication guide](https://ai.google.dev/gemini-api/docs/voice-replication)
- [Text-to-speech generation](https://ai.google.dev/gemini-api/docs/speech-generation)
