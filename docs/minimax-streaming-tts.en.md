# MiniMax WebSocket Streaming TTS

`minimax_streaming_tts` implements MiniMax's synchronous Text-to-Audio WebSocket protocol. Callers can send text in segments and receive hexadecimal audio increments as they arrive. This is a text-to-audio task stream, not a speech-to-speech realtime conversation.

Enable the `realtime-websocket` feature to use the built-in Rustls WebSocket transport, or inject a host-provided `RealtimeTransport`.

```rust,ignore
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use std::sync::Arc;
use lingxi_llm_client::{
    providers::minimax::streaming_tts::{
        MiniMaxStreamingTtsConfig, MiniMaxStreamingTtsEvent, MiniMaxStreamingTtsLimits,
        MiniMaxStreamingTtsRequest, MiniMaxStreamingTtsService,
    },
    providers::minimax::tts::MiniMaxTtsRegion,
    protocol::Secret,
    realtime::{RealtimeTransport, HttpTransport},
};

let transport: Arc<dyn RealtimeTransport> = Arc::new(HttpTransport::new()?);
let service = MiniMaxStreamingTtsService::new(
    transport,
    MiniMaxStreamingTtsConfig::new(
        "minimax-voice",
        "team/account-17",
        MiniMaxTtsRegion::International,
    ),
)?;
let credential = Secret::new(api_key);
let session = service
    .connect(
        &credential,
        &MiniMaxStreamingTtsRequest::new("English_expressive_narrator"),
        MiniMaxStreamingTtsLimits::default(),
    )
    .await?;
let (mut input, mut events) = session.into_parts();

input.send_text("The first sentence.").await?;
while let Some(event) = events.next().await? {
    if let MiniMaxStreamingTtsEvent::AudioDelta { audio, is_final, .. } = event {
        // Pass `audio` to the caller's playback or storage layer.
        let _ = audio;
        if is_final {
            break;
        }
    }
}
input.finish().await?;
while let Some(event) = events.next().await? {
    if matches!(event, MiniMaxStreamingTtsEvent::TaskFinished { .. }) {
        break;
    }
}
# Ok(())
# }
```

The service binds to a profile, account scope, and explicit region. `International` selects MiniMax's documented international WSS endpoint, `wss://api.minimax.io/ws/v1/t2a_v2`; `ChinaMainland` selects the mainland endpoint embedded in its official page, `wss://api.minimax.cn/ws/v1/t2a_v2`. The constructor chooses by region, and the service rejects an endpoint that does not exactly match the selected region. Pass `&Secret<String>` for each connection; the service does not retain credentials.

A single `MiniMaxVoiceRef` from `minimax_voices` can be passed to `connect_with_voice`. Before opening WSS, this method checks the provider, profile, account scope, region, and voice API endpoint: international streams accept only `https://api.minimax.io/v1`, and mainland streams only `https://api.minimax.cn/v1`. Cross-region references fail before network setup. This helper cannot be combined with `timbre_weights`; use `connect` to provide a complete voice mix. Account scope is a caller-supplied routing label, not API-key ownership proof; pass the key for the intended account on each connection. Built-in voice IDs remain available through `connect`.

The handshake waits for `connected_success`, sends `task_start`, and then waits for `task_started`. Both ordinary regional T2A WebSocket schemas list eight models: `speech-2.8-hd/turbo`, `speech-2.6-hd/turbo`, `speech-02-hd/turbo`, and `speech-01-hd/turbo`. The typed request covers every `task_start` field: `voice_setting` exposes voice, speed, volume, pitch, emotion, English normalization, and LaTeX reading; `audio_setting` supports MP3, WAV, FLAC, PCM, μ-law, and Opus; optional controls include pronunciation rules, voice mixing, voice effects, subtitles, language boost, and `continuous_sound`.

The client validates documented combinations before connecting: `fluent` and `whisper` emotion values are limited to `speech-2.6-hd/turbo`; `continuous_sound` is limited to `speech-2.8-hd/turbo`; `voice_modify` is documented only for streaming MP3; voice mixes allow at most four voices with weights from 1 to 100, and require an empty `voice_setting.voice_id`. `latex_read` supports Chinese only; if the caller also supplies `language_boost`, it must be `Chinese`. When that field is omitted, the client leaves it omitted as MiniMax's docs say the service sets it to Chinese. The speech-01/02 families do not support Persian, Filipino, or Tamil `language_boost`. Supported sample rates are 8, 16, 22.05, 24, 32, and 44.1 kHz; channel count is 1 or 2. MP3 bitrates are 32, 64, 128, or 256 kbit/s, and μ-law formats require 8 kHz. Audio defaults to 32 kHz, 128 kbit/s, and mono. Emotion values are `happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`, `calm`, `fluent`, and `whisper`; `voice_modify.sound_effects` accepts `spacious_echo`, `auditorium_echo`, `lofi_telephone`, and `robotic`; subtitle types are `sentence`, `word`, and `word_streaming`. Each `task_continue` sends one text segment. Audio frames decode `data.audio` from hex and preserve the complete native event, `extra_info`, session ID, trace ID, and `is_final` flag; Opus chunks must be reassembled in arrival order before decoding.

The caller can queue multiple text segments up to the configured pending limit while receiving audio through the separate event half. Default local limits are 16 KiB per text segment, 8 pending segments, 1 MiB per WebSocket message, and 64 MiB of decoded audio per session. These are client-side bounds, not MiniMax service limits. `task_finish` asks MiniMax to drain queued work and close the task; continue reading until `TaskFinished`.

The service does not reconnect or replay text. If a `task_continue` or `task_finish` send fails after it may have reached MiniMax, the result is `OutcomeUnknown`; do not automatically submit the same text again. Cancelling a `send_text` or `finish` future during a write invalidates the sender and drops the transport write half; later sends return `Closed`. Dropping an unpolled future does not invalidate the session. A disconnect before `task_finished` returns `Interrupted`, and previously yielded audio chunks remain with the caller. Dropping the session halves releases the injected transport; `abort` sends a WebSocket close for explicit cancellation.

The contract follows MiniMax's [international ordinary T2A WebSocket reference](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket) and [mainland ordinary T2A WebSocket reference](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket). Both pages provide the full regional WSS endpoint and document Bearer authorization, the `connected_success` / `task_started` handshake, `task_continue` / `task_finish`, hexadecimal audio, `is_final`, queued-task behavior, and terminal events. This implements ordinary one-way T2A WebSocket, not the separate Bidi WebSocket protocol.
