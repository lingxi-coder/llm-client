# Audio: file transcription, translation, and TTS

[简体中文](audio.md)

Bind `client.provider::<OpenAiClient>(profile)?` to an exact profile, then use `provider.audio()`. Each operation takes `RequestOptions`; the client retains no credential.

`provider.audio()` is an independent OpenAI [file transcription](https://developers.openai.com/api/docs/guides/speech-to-text), [translation](https://developers.openai.com/api/reference/python/resources/audio/subresources/translations/methods/create), and [text-to-speech](https://developers.openai.com/api/docs/guides/text-to-speech) service, separate from Chat. The built-in `openai` profile configures independent routes; configuration v3 can inherit or disable them. Audio bytes use fixed-length multipart streaming, without whole-file buffering fallback. The input must be a completed recording of 1–25,000,000 bytes with a supported extension and MIME type.

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

For larger files, construct `AudioInput { filename, media_type, size_bytes, body }` directly. `body` is a one-shot `BoxStream<Result<Bytes, LlmError>>`. The declared length must match the actual stream. A custom `Transport` without streaming request support rejects the operation before consuming input. A total deadline covers the request; the client does not retry or fail over. An interrupted upload may already have been accepted and returns `OutcomeUnknown`.

`TranscriptionRequest` supports `gpt-transcribe`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-mini-transcribe-2025-12-15`, `gpt-4o-transcribe-diarize`, and `whisper-1`. This slice permits JSON only for the first four. Whisper supports text, SRT, VTT, JSON, and verbose JSON; word or segment timestamps require verbose JSON. The diarization model can return `segments[].speaker` in `diarized_json`. For audio longer than 30 seconds, callers must enable `chunking_auto`; the client does not guess the duration of a byte stream. Unsupported model/format combinations fail before sending. `translate()` uses only the documented `whisper-1` model and translates speech into English, with text, subtitle, or JSON output. `AudioTextResult` preserves full native JSON, text, a single `language` or `gpt-transcribe`'s detected `languages[]`, duration, word/segment timestamps, and speaker fields when returned.

To map segments to known speakers, add up to four `KnownSpeakerReference` values to a diarization request. Each contains a speaker name, supported-format filename and MIME type, audio bytes, and caller-supplied duration. The client requires durations from 2 to 10 seconds and encodes each pair as `known_speaker_names[]` and `known_speaker_references[]` multipart fields; each reference is sent as a `data:{mime};base64,...` URL. References are accepted only with `gpt-4o-transcribe-diarize` and `diarized_json`. The caller supplies the actual clip duration; the client does not decode audio to estimate it. The same fields are available through `transcribe_stream()`, where the provider assigns a speaker only when it finalizes a segment. See OpenAI's [speaker diarization guide](https://developers.openai.com/api/docs/guides/speech-to-text#speaker-diarization).

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

`synthesize()` takes text, model, voice, output format, and optional speed. It returns a `SpeechStream` read in binary chunks; it does not write a file or play audio. Supported models are `gpt-4o-mini-tts`, its listed dated version, `tts-1`, and `tts-1-hd`. The older TTS models have a smaller voice set and do not support `instructions`. Input is limited to 1–4096 characters and speed to 0.25–4.0. Output formats are MP3, Opus, AAC, FLAC, WAV, and PCM. PCM is raw 24 kHz, signed 16-bit little-endian mono. A stream interruption reports the number of bytes already delivered; the client never resubmits automatically. Apps playing TTS to end users should follow the [official disclosure requirement](https://developers.openai.com/api/docs/guides/text-to-speech) that the voice is AI-generated.

Custom voice selection requires an opaque `CustomVoiceRef`, not a bare voice ID. `create_voice()` returns `CustomVoice` metadata; pass `voice.reference().clone()` to `SpeechVoice::Custom`. To use a previously approved ID, call `provider.audio().voices().import_approved_voice(id, options)`. Import is local and requires an explicit profile plus `RequestOptions.account_scope`; it does not check whether the ID exists or is authorized. A reference is serializable with its provider, profile, official Speech endpoint fingerprint, and caller-declared account scope, while the HTTP request still sends only `"voice": {"id":"voice_123abc"}`. Before reading the credential or sending HTTP, synthesis rejects an invalid ID or a provider, profile, Speech endpoint, or account-scope mismatch. This scope check cannot establish project authorization, consent, or whether a voice remains available. Create and consent-recording lifecycle operations are available through `provider.audio().voices()` below. See [OpenAI custom voices](https://developers.openai.com/api/docs/guides/custom-voices).

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

`provider.audio().voices()` exposes the documented consent phrase catalog, consent recording creation and CRUD, and custom voice creation. Voice creation currently has a create endpoint only, so this client does not invent list, retrieve, update, or delete operations for voices. The phrase catalog is returned as raw JSON because OpenAI documents its endpoint without a response schema. The client derives these resource paths from the selected OpenAI audio route and uses that profile's bearer credential. Keep the same project-scoped API key and profile for every lifecycle request; also set the same stable, non-secret `RequestOptions.account_scope` on each call. Returned `VoiceConsentRef` values bind the consent ID to that profile, endpoint, and caller-declared account scope. Get, update, delete, and voice creation reject a mismatch before HTTP. This check compares the caller-declared scope; it cannot prove that the supplied key belongs to the named project. Reading phrases requires `api.voices.read`, while creating consents and voices requires `api.voices.write` and custom-voice access. Custom voices are available only to eligible customers.

Consent and sample recordings must be separate recordings from the same speaker. The client streams the documented multipart fields and requires each upload to be nonempty and at most 10 MiB, with one of OpenAI's supported audio MIME types (`audio/mpeg`, `audio/wav`, `audio/x-wav`, `audio/ogg`, `audio/aac`, `audio/flac`, `audio/webm`, or `audio/mp4`). OpenAI limits samples to 30 seconds and requires the consent recording to contain exactly a supported consent phrase; the provider validates recording content and access. The voice consent API exposes list, retrieve, metadata-only name update, and delete with a deletion receipt. Pagination is caller-controlled; the client does not fetch subsequent pages automatically.

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
    // Set options.account_scope to a stable, non-secret ID for this OpenAI project.
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
        input: "Hello".into(),
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

File transcription can also call `provider.audio().transcribe_stream(input, &request, options)`. It sends `stream=true` and returns native `transcript.text.delta`, `transcript.text.segment` (diarization), and terminal `transcript.text.done` events one by one. Use the done event for the complete text. This API accepts GPT transcription models with JSON or diarized JSON output and rejects `whisper-1`. `TranscriptionEventStream::next_event()` reports a missing terminal event, transport interruption, or invalid event as an error. The upload is one-shot; the client does not resubmit or resume it. See the [OpenAI Docs file transcription guide](https://developers.openai.com/api/docs/guides/speech-to-text).

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

`synthesize_stream()` can read the same Speech request as SSE events. It is limited to GPT TTS models on the official OpenAI `/v1/audio/speech` route; the API reference excludes `tts-1` and `tts-1-hd` from SSE. It decodes Base64 from `speech.audio.delta.audio` and preserves `speech.audio.done.usage` as native JSON. A valid terminal `done` must include integer `input_tokens`, `output_tokens`, and `total_tokens`. Unrecognized events, including undocumented in-stream errors, retain their `event_type`, native JSON, and raw SSE data; the client does not infer error fields. Each SSE event is capped at 8 MiB. A transport interruption or EOF without a valid terminal event returns `SpeechEventStreamError::Interrupted`; missing documented fields or invalid Base64 on a known event returns `InvalidEvent`. Non-2xx HTTP statuses remain `AudioError::Provider`. As of 2026-09-27, the reviewed OpenAI Speech documentation does not establish an SSE text/audio alignment contract, so this client does not infer alignment fields; this is not a claim that the service can never support them. This Speech API does not include asynchronous audio jobs, Chat audio, or Realtime; those capabilities belong to separate services. No live OpenAI account was called.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::audio::{SpeechEvent, SpeechModel, SpeechRequest, SpeechVoice};

async fn speak_events(client: &LlmClient, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let request = SpeechRequest {
        model: SpeechModel::Gpt4oMiniTts,
        input: "Hello".into(),
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
