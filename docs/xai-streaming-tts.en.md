# xAI Streaming Text to Speech

`xai_streaming_tts` implements xAI's bidirectional streaming text-to-speech WebSocket protocol at the fixed `wss://api.x.ai/v1/tts` route. It is a text-to-audio stream, separate from xAI's speech-to-speech realtime conversation and the HTTP TTS API. The host provides a `RealtimeTransport` and an API key for each connection.

```rust,no_run
use futures::future::join;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{RealtimeError, RealtimeLimits, RealtimeTransport},
    providers::xai::streaming_tts::{
        XaiStreamingTtsConfig, XaiStreamingTtsEvent, XaiStreamingTtsSession,
    },
};
use std::{error::Error, sync::Arc};

async fn synthesize_one(
    transport: Arc<dyn RealtimeTransport>,
    api_key: String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (session, driver) = XaiStreamingTtsSession::connect(
        transport,
        Secret::from(api_key),
        XaiStreamingTtsConfig::new("en"),
        RealtimeLimits::default(),
    )
    .await?;
    let (control, mut events) = session.into_parts();
    control.send_text_delta("Hello from a streaming TTS session.")?;
    control.finish_utterance()?;

    let close_control = control.clone();
    let receive = async move {
        while let Some(event) = events.next().await {
            match event {
                XaiStreamingTtsEvent::AudioDelta { audio, .. } => {
                    // Store or forward these codec bytes in the host.
                    let _ = audio;
                }
                XaiStreamingTtsEvent::AudioDone { .. } => {
                    close_control.close().await?;
                    return Ok(());
                }
                XaiStreamingTtsEvent::ProviderError { message, .. } => {
                    close_control.close().await?;
                    return Err(RealtimeError::Codec { message });
                }
                XaiStreamingTtsEvent::ConnectionInterrupted { message } => {
                    return Err(RealtimeError::Transport { message });
                }
                _ => {}
            }
        }
        Err(RealtimeError::UnexpectedRemoteClose)
    };

    let (driver_result, receive_result) = join(driver.run(), receive).await;
    driver_result?;
    receive_result?;
    Ok(())
}
```

The WebSocket upgrade is the readiness signal; the server does not send a `created` or `ready` event. `XaiStreamingTtsConfig::new(language)` sets a required language and otherwise uses the default voice, MP3 format, 24 kHz sample rate, 128 kbit/s bitrate, speed 1.0, and latency optimization level 0. Set `voice` to select a built-in or custom voice. Supported codecs are MP3, WAV, PCM, μ-law, and A-law. Sample rates are 8000, 16000, 22050, 24000, 44100, or 48000 Hz; `bit_rate` is only accepted for MP3 at 32000, 64000, 96000, 128000, or 192000 bit/s. Speed is 0.7–1.5. The WebSocket guide documents `optimize_streaming_latency` values 0, 1, and 2. `text_normalization` and `with_timestamps` are optional booleans.

Call `send_text_delta` with nonempty text chunks of at most 60,000 Unicode characters. Call `finish_utterance` once to send `text.done`; then consume `AudioDelta` events until `AudioDone`. Audio deltas contain decoded raw bytes in the selected codec. Keep the driver running while sending and receiving; the event and command queues and each frame are bounded by `RealtimeLimits`. If the local queue is full, the command returns `QueueFull` and does not advance the utterance state.

One WebSocket can carry multiple utterances. `AudioDone` marks the end of one audio response; it does not close the connection. To cancel, call `clear_utterance`, discard any buffered audio at the host, and wait for `AudioClear` before sending more text. A `session.update` replacement map is queued with `update_replacements`; changes take effect on the next utterance, and `SessionUpdated` reports the provider acknowledgement. This map controls provider-side phrase replacement; it does not rewrite the text sent by the client.

Known audio, clear, and session-update events are typed when their documented fields are valid. Unknown or malformed events remain available through `ProviderEvent`; for example, an `audio.done` without a valid `trace_id` is retained raw and does not complete the utterance. Provider `error` messages are surfaced as `ProviderError`. Remote EOF, transport failure, and provider connection loss are interruptions rather than successful completion; malformed events remain raw and do not advance session state. The driver never reconnects or replays text. Call `close` to close the WebSocket locally; that is distinct from `finish_utterance` and does not mean synthesis completed. Audio decoding beyond Base64 transport decoding, buffering, playback, and device access remain the host's responsibility.

The API contract follows xAI's [Streaming Text to Speech guide](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech) and its [voice inference reference](https://docs.x.ai/developers/rest-api-reference/inference/voice). The WebSocket route uses query options and JSON text messages; do not send the HTTP route's JSON body to it. The built-in Rustls WebSocket transport is available with the `realtime-websocket` feature; the example above uses the default-feature injected-transport API.
