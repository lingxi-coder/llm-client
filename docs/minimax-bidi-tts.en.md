# MiniMax Bidirectional Streaming TTS

`minimax_bidi_tts` implements MiniMax's native `t2a_v2_bidi` bidirectional WebSocket lifecycle. The International route is `wss://api.minimax.io/ws/v1/t2a_v2_bidi`; the Mainland route is `wss://api.minimax.cn/ws/v1/t2a_v2_bidi`. The service takes an injected `RealtimeTransport`, receives credentials per connection, and selects the official route from an explicit region.

```rust,no_run
# async fn example(
#     transport: std::sync::Arc<dyn lingxi_llm_client::realtime::RealtimeTransport>,
#     api_key: String,
# ) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    minimax_bidi_tts::{
        MiniMaxBidiTtsConfig, MiniMaxBidiTtsCredentials, MiniMaxBidiTtsError,
        MiniMaxBidiTtsEventKind, MiniMaxBidiTtsLanguageBoost, MiniMaxBidiTtsLimits,
        MiniMaxBidiTtsParameters, MiniMaxBidiTtsRegion, MiniMaxBidiTtsRequest,
        MiniMaxBidiTtsService,
    },
    protocol::Secret,
};

let service = MiniMaxBidiTtsService::new(
    transport,
    MiniMaxBidiTtsConfig::new(
        "voice-profile",
        "team/account-17",
        MiniMaxBidiTtsRegion::ChinaMainland,
    ),
)?;
let credentials = MiniMaxBidiTtsCredentials::new(Secret::new(api_key));
let mut parameters = MiniMaxBidiTtsParameters::new("English_expressive_narrator");
parameters.language_boost = Some(MiniMaxBidiTtsLanguageBoost::Chinese);
let request = MiniMaxBidiTtsRequest::new("English_expressive_narrator")
    .with_parameters(parameters)
    .with_session_id("host-session-17");

// connect opens the session and sends task_start.
let session = service
    .connect(&credentials, &request, MiniMaxBidiTtsLimits::default())
    .await?;
let (mut input, mut events) = session.into_parts();
input.continue_text("Hello from a streaming text source.").await?;
// Host-scheduled keepalive only; the injected transport may report unsupported.
match input.ping(Vec::<u8>::new().into()).await {
    Ok(()) | Err(MiniMaxBidiTtsError::PingUnsupported) => {}
    Err(error) => return Err(error.into()),
}
input.finish().await?;
let mut task_finished_seen = false;
while let Some(event) = events.next().await? {
    match event.kind {
        MiniMaxBidiTtsEventKind::AudioDelta { audio, is_final, .. } => {
            let _audio_chunk = audio;
            let _request_audio_is_final = is_final;
        }
        MiniMaxBidiTtsEventKind::TaskFinished => task_finished_seen = true,
        MiniMaxBidiTtsEventKind::TaskFailed { .. } => {
            return Err(std::io::Error::other("MiniMax task failed").into());
        }
        MiniMaxBidiTtsEventKind::SoftError { .. } => {
            // The host decides whether to retry; this client never replays text.
        }
        _ => {}
    }
}
if !task_finished_seen {
    return Err(std::io::Error::other("session ended without task_finished").into());
}
# Ok(())
# }
```

`MiniMaxBidiTtsParameters` reuses the ordinary MiniMax WebSocket T2A typed task settings; Bidi adds its optional client-generated `session_id`. The official schema lists eight models, nested voice and audio settings, pronunciation dictionaries, mixed timbres, voice modification, subtitles, language boost, and `continuous_sound`. Shared validation checks the documented setting combinations without applying ordinary WSS route restrictions.

Connection setup waits for `connected_success`, sends one `task_start`, and then waits for `task_started` before returning a session. The connect deadline covers the handshake and both readiness events. A failure before `task_start` is sent is a connection or protocol error. If a task-start write is attempted but its acknowledgement is not confirmed, the result is reported as unknown. The client never retries or reconnects.

Send each `task_continue` text fragment as it arrives. The service does not trim, normalize, add punctuation, or split text. It preserves empty strings, whitespace, newlines, and punctuation; MiniMax silently drops whitespace-only text. The provider accepts arbitrary text granularity and buffers it into sentences. A fragment can contain up to 10,000 Unicode characters; the client applies the same default local cap and checks serialized frame size before sending.

The event stream keeps three completion levels distinct: audio `is_final`, sentence `sentence_end`, and session `task_finished`. Audio arrives as hex in `data.audio`; the client decodes each chunk to bytes and retains its native JSON, `extra_info`, `session_id`, `trace_id`, and `connect_id`. Sentence boundary events remain separate. Unknown JSON events are returned with their full native value; malformed decoded events preserve that value, and invalid JSON or binary frames retain their exact bytes in `raw_frame`.

`input.flush()` asks MiniMax to synthesize buffered tail text while keeping the session open. Read through `TaskFlushed` before sending more text. `input.cancel()` interrupts active synthesis, including a pending flush; read `TaskCanceled` before continuing. A late `TaskFlushed` received while cancellation is pending is surfaced but does not release the cancel gate. Neither control method retries or resends content.

The input and event halves can make progress concurrently. A control acknowledgement that arrives before its outbound write finishes is retained, but the input gate opens only after the write itself succeeds. If that write fails, the client reports an unknown result and does not claim the operation succeeded.

`input.finish()` asks MiniMax to synthesize remaining text and close the task. Its send result alone is not completion. For successful completion, the client must receive a valid `TaskFinished` event and then a clean WebSocket EOF. A `TaskFailed` event can also be followed by `events.next()` returning `None`; callers must treat that as failure, not success. EOF before `TaskFinished` and transport errors after it are surfaced as errors. `abort()` closes locally without claiming task completion.

If a `continue_text`, `flush`, `cancel`, or `finish` future is dropped while its frame write is in flight, the client fails the session closed because it cannot know whether MiniMax received the frame. Further input returns `Closed`; it is never silently retried or made ready again. Call `abort()` to release the session explicitly. A fatal provider `task_failed` event closes the WebSocket; the event is surfaced once and the event stream then terminates.

Provider codes 2204 and 2205 are exposed as `SoftError` events and do not close the session. The host chooses whether to send text again; the client never replays it. Other nonzero provider status codes and `task_failed` events are terminal failures. The official guide says 2204 skips a text fragment that exceeds the character limit, while 2205 means the service queue contains too much pending synthesis text.

MiniMax may close a connection after about 120 seconds of inactivity. The host can schedule `input.ping(payload)` to send an actual WebSocket Ping control frame with a payload no larger than 125 bytes; it is never encoded as JSON. The client does not provide an automatic ping timer or reconnect. The injected transport must support explicit Ping; the default implementation reports `PingUnsupported`, and the application owns keepalive timing.

`connect_with_voice` accepts a `MiniMaxVoiceRef` only when provider, profile, account scope, region, and official HTTP `/v1` root match this service. It checks the reference before opening WSS and uses its voice ID in `task_start`. Account scope is caller-provided metadata, not proof that an API key belongs to that account.

`MiniMaxBidiTtsLimits` bounds inbound/outbound frames, text characters, cumulative returned audio bytes, and unknown setup events. These are local memory and safety limits; only the 10,000-character text limit is also documented by MiniMax. The client does not decode codecs, assemble Opus containers, play audio, or choose keepalive intervals.

Official references: [International Bidi T2A WebSocket](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket-bidi), [Mainland Bidi T2A WebSocket](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket-bidi), [MiniMax voice lifecycle](minimax-voices.en.md), and [MiniMax ordinary WebSocket TTS](minimax-streaming-tts.en.md).
