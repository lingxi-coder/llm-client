# xAI Audio Service

`xai_audio` provides xAI-native HTTP speech-to-text and text-to-speech operations, plus a read-only TTS voice catalog. It does not reuse OpenAI audio fields: STT uploads multipart audio and returns JSON; ordinary TTS returns a raw audio byte stream; requesting character timestamps changes TTS to a JSON envelope whose `audio` field contains base64 audio.

```rust,no_run
# async fn example(api_key: String, audio_bytes: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    audio::AudioInput,
    HttpTransport,
    protocol::Secret,
    xai_audio::{
        XaiAudioConfig, XaiAudioCredentials, XaiAudioService, XaiSpeechOutput,
        XaiSpeechRequest, XaiTranscriptionRequest, XaiTranscriptionUrl,
    },
};

let transport = HttpTransport::new()?;
let service = XaiAudioService::new(
    &transport,
    XaiAudioConfig::new("xai-prod", "team/account-7"),
)?;
let credentials = XaiAudioCredentials::new(Secret::new(api_key));

let voices = service.list_voices(&credentials).await?;
for voice in &voices.voices {
    println!("{} ({})", voice.name, voice.voice_id);
}

let transcript = service
    .transcribe(
        AudioInput::from_bytes("meeting.wav", "audio/wav", audio_bytes),
        &XaiTranscriptionRequest::default(),
        &credentials,
    )
    .await?;
println!("{}", transcript.text);

let audio_url = XaiTranscriptionUrl::new(
    "https://media.example/meeting.wav?signature=example",
)?;
let remote_transcript = service
    .transcribe_url(&audio_url, &XaiTranscriptionRequest::default(), &credentials)
    .await?;
println!("{}", remote_transcript.text);

let mut speech_request = XaiSpeechRequest::new("Welcome to Acme Mobile.", "eve", "en");
speech_request.speed = Some(1.2);
speech_request.optimize_streaming_latency = Some(1);
speech_request.text_normalization = true;
speech_request
    .replace
    .insert("Acme Mobile".into(), "Acme Mobull".into());
let output = service.synthesize(&speech_request, &credentials).await?;
if let XaiSpeechOutput::Audio(mut audio) = output {
    let mut audio_file = Vec::new();
    while let Some(chunk) = audio.next_chunk().await? {
        audio_file.extend_from_slice(&chunk);
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
# }
```

`XaiAudioConfig` requires a nonempty profile name and stable account scope. It defaults to `https://api.x.ai/v1`; `with_api_base_url` selects an explicit compatible HTTPS API root. HTTP is accepted only for localhost/loopback test transports. Each call receives an `XaiAudioCredentials` value containing the caller-owned API key. The service and results do not store credentials. Results include the profile, account scope, and a fingerprint of the configured API root.

## Speech to text

`XaiAudioService::transcribe` calls `POST /v1/stt` with a streamed `multipart/form-data` upload. xAI requires all option fields before the final `file` field; the implementation preserves that order without buffering the upload. The documented upload limit is 500 MB. Supported containers include WAV, MP3, OGG, Opus, FLAC, AAC, MP4/M4A, and MKV. Raw PCM, μ-law, or A-law input requires `audio_format` and a supported `sample_rate`.

For server-readable audio, construct `XaiTranscriptionUrl` and call `XaiAudioService::transcribe_url`. The client puts the source in the documented multipart `url` field for xAI to download and transcribe; the client does not download the media. The caller must ensure xAI can reach the HTTP(S) URL. Signed query parameters are preserved verbatim. Construction rejects non-HTTP(S) URLs, user information, and fragments before sending a request. `XaiTranscriptionUrl` redacts its `Debug` output. Error cleanup replaces only the exact full input URL or exact full `?query` fragment when echoed verbatim; it does not guarantee cleanup if the provider reorders, normalizes, or partially echoes the URL. Callers must decide whether to log native error content.

The typed request covers the current model IDs, language formatting, multichannel transcription, diarization, keyterms, filler words, and VAD threshold. `format=true` requires a language. Raw audio requires an explicit format and sample rate. The result retains transcript text, detected language, duration, word timestamps, speaker IDs, per-channel output, request ID, and native JSON. Malformed transcript fields become invalid-response errors.

## Text to speech

`XaiAudioService::synthesize` calls `POST /v1/tts` with xAI's JSON fields: `text`, `voice_id`, `language`, and `output_format`. The default is MP3 at 24 kHz / 128 kbps. WAV, PCM, μ-law, and A-law codecs are supported by the request type; sample rates and MP3 bit rates are checked against xAI's published values. Text is limited to 60,000 characters.

`XaiSpeechRequest` also exposes `speed` (0.7–1.5, default 1.0), `optimize_streaming_latency` (0, 1, or 2; default 0), and `text_normalization` (default false). `None`/false values are omitted from the JSON request, preserving provider defaults. Latency level 1 or 2 reduces first-chunk size for streaming synthesis, with an audio-quality tradeoff at chunk boundaries.

The optional `replace` map changes spoken substitutions without changing the original text sent or billed. The client validates the documented map limits before sending: at most 200 entries, keys up to 100 Unicode code points, values up to 128 code points, and keys containing only letters, digits, apostrophes, or spaces. Keys must be nonblank and distinct after standard lowercase conversion and whitespace removal. This local duplicate check does not implement full Unicode case folding; xAI remains authoritative for matching and validation. xAI matches case-insensitively with whole-word boundaries, applies the longest matching key, and treats scripts without spaces as per-character matches. The provider enforces the 240,000-character post-substitution limit; the client does not emulate its matching engine. With `with_timestamps=true`, `graph_chars` describes spoken replacement text rather than the original key.

Without timestamps, `XaiSpeechOutput::Audio` yields raw HTTP audio chunks and exposes the provider content type and request ID. With `with_timestamps=true`, xAI returns JSON containing Base64 audio, content type, duration, and per-character timings; the client decodes this as `XaiSpeechOutput::Timestamped`. The two response shapes remain distinct.

## TTS voice catalog

`XaiAudioService::list_voices` calls the read-only `GET /v1/tts/voices` endpoint. Each typed voice exposes `voice_id` and `name`; `language` is optional because the TTS guide does not require it in every voice-list example. `XaiVoice.native` and `XaiVoiceList.native` preserve the full provider voice objects and response envelope, including fields this client does not interpret. The returned list is bound to the service profile, account scope, and endpoint fingerprint. This method lists voices available for TTS; it does not create or manage custom voice resources.

`XaiAudioService::get_voice(voice_id, credentials)` calls `GET /v1/tts/voices/{voice_id}` and returns `XaiVoiceDetails`, which carries the service scope, a typed `XaiVoice`, the complete native response, and the request ID. The returned `voice_id` must exactly match the requested ID. The ID is appended as one path segment; empty/trimmed values, controls, separators, percent escapes, and dot segments are rejected before HTTP. The method is read-only and makes one request without retrying.

These HTTP operations do not look up credentials, retry, fail over across accounts, or cache results. A transport failure after sending a transcription or synthesis request can leave the provider-side outcome unknown; such errors use `XaiAudioError::OutcomeUnknown`.

Implementation follows xAI's [Speech to Text guide](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text), [Text to Speech guide](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech), and [Voice REST reference](https://docs.x.ai/developers/rest-api-reference/inference/voice).

Custom voice creation, management, and reference-audio streaming are available through the same service; see [Custom Voices](xai-custom-voices.en.md).
