# xAI Realtime Speech to Text

`xai_stt::XaiSttSession` implements xAI's transcription-only WebSocket protocol at the fixed `wss://api.x.ai/v1/stt` route. It uses the shared bounded realtime transport, while keeping its binary audio and transcript-event contract separate from xAI's speech-to-speech conversation API. The host supplies an `Arc<dyn RealtimeTransport>` and a request-scoped API key. See xAI's [Speech to Text guide](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text) for the current wire contract.

```rust,no_run
use futures::future::join;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{RealtimeError, RealtimeLimits, RealtimeTransport},
    xai_stt::{XaiSttConfig, XaiSttEvent, XaiSttSession},
};
use std::{error::Error, sync::Arc};

async fn transcribe(
    transport: Arc<dyn RealtimeTransport>,
    api_key: String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (session, driver) = XaiSttSession::connect(
        transport,
        Secret::from(api_key),
        XaiSttConfig::default(),
        RealtimeLimits::default(),
    )
    .await?;
    let (control, mut events) = session.into_parts();

    // Supply raw PCM16 bytes in suitably paced chunks in a real application.
    control.send_audio(vec![0_u8, 0])?;
    control.finish_audio()?;

    let (driver_result, receive_result) = join(driver.run(), async move {
        while let Some(event) = events.next().await {
            if matches!(event, XaiSttEvent::Completed { .. }) {
                break;
            }
        }
        Ok::<(), RealtimeError>(())
    })
    .await;
    driver_result?;
    receive_result?;
    Ok(())
}
```

`connect` waits for `transcript.created` before returning and applies `connect_timeout` to both the WebSocket handshake and that readiness event. `XaiSttConfig` selects `grok-voice-transcribe-1.0` or the default `grok-voice-transcribe-2.0`, raw `pcm`, `mulaw`, `alaw`, or `opus` encoding, language formatting, endpointing, diarization, filler words, multichannel input, keyterms, Smart Turn, and VAD threshold. `sample_rate_hz: None` leaves the server's 16 kHz default in effect; non-Opus sample rates can be 8000, 16000, 22050, 24000, 44100, or 48000 Hz. Opus omits the sample-rate parameter, is mono-only, and requires exactly one raw Opus packet in each binary frame.

`send_audio` sends the bytes as one binary WebSocket frame without Base64 conversion. The host owns audio capture, chunk pacing, and any resampling. `finalize_utterance` sends the channel-agnostic lowercase `{"type":"finalize"}` message while leaving input open; this follows the lowercase form in the Client Messages list. The same page's separate push-to-talk examples use `Finalize`, so this adapter does not assume both spellings are interchangeable. `finish_audio` queues `{"type":"audio.done"}` and closes the local input side only after the bounded queue accepts it. Run the returned driver while consuming events.

Typed partials include the documented text, words, finality flags, timestamps, optional channel index, and optional Smart Turn confidence. A typed `TranscriptDone` requires a finite nonnegative duration and a valid, previously unseen channel. In multichannel mode the driver considers the provider's remote EOF successful only after `audio.done` was actually sent and one valid final event arrived for every configured channel. `Completed { expected_channels }` reports that EOF; no close code is invented. A missing or malformed final, a provider error, or a transport error remains an error/event. Unrecognized or malformed provider events retain their native JSON in `ProviderEvent`.

The module does not reconnect, replay audio, capture devices, or turn this transcription stream into a voice conversation. A Rustls WebSocket transport is available behind the crate's `realtime-websocket` feature; otherwise inject the transport implementation owned by the host.
