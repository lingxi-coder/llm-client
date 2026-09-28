# MiniMax HTTP TTS

`minimax_tts` provides an independent service for MiniMax's native `POST /v1/t2a_v2` HTTP API, with both synchronous JSON output and SSE streaming. It uses MiniMax's own `voice_setting` / `audio_setting` request body and `base_resp` status instead of treating the service as OpenAI-compatible TTS.

```rust,no_run
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    providers::minimax::tts::{
        MiniMaxTtsConfig, MiniMaxTtsCredentials, MiniMaxTtsOutput, MiniMaxTtsRegion,
        MiniMaxTtsRequest, MiniMaxTtsService,
    },
    protocol::Secret,
    HttpTransport,
};

let transport = HttpTransport::new()?;
let service = MiniMaxTtsService::new(
    &transport,
    MiniMaxTtsConfig::new("minimax-voice", "team/account-17", MiniMaxTtsRegion::International),
)?;
let credentials = MiniMaxTtsCredentials::new(Secret::new(api_key));
let request = MiniMaxTtsRequest::new("Hello from MiniMax", "English_expressive_narrator");
match service.synthesize(&request, &credentials).await? {
    MiniMaxTtsOutput::Audio(audio) => {
        let mut output_file = Vec::new();
        output_file.extend_from_slice(&audio.bytes);
    }
    MiniMaxTtsOutput::Url(audio_url) => {
        println!("caller-managed URL, valid for {:?}: {}", audio_url.valid_for, audio_url.url);
    }
}
# Ok(())
# }
```

Choose `MiniMaxTtsRegion::International` for the `.io` account route or `ChinaMainland` for the `.cn` account route. The config also accepts MiniMax's documented same-region alternate endpoint. A profile name and stable account scope are required and attached to each result with an endpoint fingerprint. Pass the matching account's API key in `MiniMaxTtsCredentials` for each call; the service never stores credentials.

A `MiniMaxVoiceRef` obtained from `minimax_voices` can be passed to `synthesize_with_voice(&request, &voice, &credentials)`. Before sending HTTP, the service checks that the reference matches its provider, profile, account scope, region, and normalized `/v1` API root, then uses the referenced voice ID. Built-in voice IDs remain available through `synthesize`. The account scope is a caller-supplied routing label, not proof of credential ownership; pass the API key for the intended MiniMax account.

The request defaults to `speech-2.8-hd`, mono MP3 at 32 kHz / 128 kbps, hexadecimal output, and subtitles disabled. It supports MiniMax's listed speech model IDs, custom voice IDs, speed, volume, pitch, emotion, language boost, pronunciation entries, audio format, AIGC watermark, and the HTTP `subtitle_enable` / `subtitle_type` controls. Synchronous synthesis accepts `sentence` or `word` subtitles. `synthesize_stream` supports the documented streaming-only `word_streaming` option and requires MP3 plus hex output; URL output, WAV streaming, and AIGC watermarking are non-streaming features. Text must be nonempty and shorter than 10,000 characters. Documented input ranges are validated before sending the request.

With `MiniMaxTtsOutputFormat::Hex`, MiniMax returns `data.audio` as hexadecimal text; the service decodes it into raw codec bytes. With `MiniMaxTtsOutputFormat::Url`, MiniMax returns a provider-hosted URL valid for 24 hours. The service returns the URL and validity window without opening it or downloading audio. Results also retain `trace_id`, available `extra_info`, and the full native response.

`synthesize_stream` returns one `MiniMaxTtsStreamEvent` at a time. Each JSON event retains its full native object and decodes `data.audio` from hex when present; `data.status == 2` and `[DONE]` end the stream. A clean EOF after audio is also accepted, matching MiniMax's published CLI behavior. Check `terminal_event_received()` to distinguish an explicit status-2 or `[DONE]` marker from clean EOF; the client does not synthesize a missing terminal marker. It preserves terminal event audio and does not combine or deduplicate chunks because the provider contract available to this client does not say whether terminal audio is another delta or a complete aggregate. Streaming subtitle fields remain inside each native event; this client does not invent a subtitle event schema or download subtitle URLs.

```rust,no_run
use lingxi_llm_client::providers::minimax::tts::{
    MiniMaxTtsCredentials, MiniMaxTtsRequest, MiniMaxTtsService,
    MiniMaxTtsStreamEvent, MiniMaxTtsSubtitleType,
};

async fn collect_audio(
    service: &MiniMaxTtsService<'_>,
    credentials: &MiniMaxTtsCredentials,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
let mut audio_output = Vec::new();

let mut request = MiniMaxTtsRequest::new("Hello from a stream", "voice-id");
request.subtitle_enable = true;
request.subtitle_type = MiniMaxTtsSubtitleType::WordStreaming;
let mut events = service.synthesize_stream(&request, credentials).await?;
while let Some(event) = events.next_event().await? {
    match event {
        MiniMaxTtsStreamEvent::Data(data) => {
            if let Some(audio_chunk) = data.audio {
                audio_output.extend_from_slice(&audio_chunk);
            }
            // Inspect `data.status` and `data.native` before deciding whether
            // terminal-event audio should be appended by your application.
        }
        MiniMaxTtsStreamEvent::Done => break,
    }
}
Ok(audio_output)
}
```

MiniMax's WebSocket and `t2a_async_v2` task APIs are separate protocols; this module does not start or poll asynchronous tasks, retry requests, or download returned audio or subtitle URLs. Dropping an SSE stream cancels its HTTP body. SSE events are capped at 8 MiB each, and stream errors retain the request ID and amount of decoded audio already delivered.

The wire contract follows MiniMax's [Text to Speech (T2A) HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http) and its [mainland China HTTP reference](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http).
