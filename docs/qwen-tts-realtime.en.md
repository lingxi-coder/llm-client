# Qwen standalone TTS Realtime

`qwen_tts_realtime` implements Qwen-TTS Realtime's native WebSocket protocol: incremental text input, streamed audio, buffer commit/clear, and graceful session finish. It uses the injectable `RealtimeTransport`; the optional `realtime-websocket` feature provides `RustlsWebSocketTransport`. There is no background driver, implicit reconnect, text replay, audio playback, or credential refresh.

The route is selected explicitly:

| Region | Endpoint before the model query |
| --- | --- |
| Beijing | `wss://dashscope.aliyuncs.com/api-ws/v1/realtime` |
| Singapore | `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime` |

The service adds `?model=...`, `Authorization: Bearer ...`, and `X-DashScope-WorkSpace`. `QwenTtsRealtimeConfig` binds profile, non-secret account identity, region and workspace; `scope()` also exposes the endpoint fingerprint. Each connection takes credentials through `RequestOptions`. If `account_scope` is supplied there, it must match the service. The API key must belong to the selected region/account; the client does not infer this from a Chat profile. [Official connection guide](https://help.aliyun.com/en/model-studio/interactive-process-of-qwen-tts-realtime-synthesis).

## Models, voices and configuration

The current documented model IDs are accepted explicitly. Unknown models are rejected before the handshake.

| Family | Accepted model IDs | Regions / voice origin |
| --- | --- | --- |
| Flash | `qwen3-tts-flash-realtime`, `qwen3-tts-flash-realtime-2025-11-27`, `qwen3-tts-flash-realtime-2025-09-18` | Both / `System` |
| Instruct | `qwen3-tts-instruct-flash-realtime`, `qwen3-tts-instruct-flash-realtime-2026-01-22` | Both / `System` |
| Voice cloning | `qwen3-tts-vc-realtime-2026-01-15`, `qwen3-tts-vc-realtime-2025-11-27` | Both / `Cloned` |
| Voice design | `qwen3-tts-vd-realtime-2026-01-15`, `qwen3-tts-vd-realtime-2025-12-16` | Both / `Designed` |
| Legacy | `qwen-tts-realtime`, `qwen-tts-realtime-latest`, `qwen-tts-realtime-2025-07-15` | Beijing only / `System` |

`QwenTtsRealtimeVoice` makes the voice origin explicit. The client checks the model family against that origin; it does not create custom voices or prove account ownership. Individual voice IDs and voice-language availability remain provider-authoritative. [Model and region list](https://www.alibabacloud.com/help/en/model-studio/realtime-tts-user-guide), [Flash model card](https://help.aliyun.com/en/model-studio/qwen3-tts-flash-realtime), [Instruct model card](https://help.aliyun.com/en/model-studio/qwen3-tts-instruct-flash-realtime).

The request defaults to `ServerCommit`, language `Auto`, PCM at 24 kHz. Qwen3 families support PCM/WAV/MP3/Opus, 8/16/24/48 kHz, speech and pitch rates 0.5–2.0, volume 0–100, and Opus bitrate 6–510 kbps. Legacy models accept only PCM at 24 kHz and reject those optional rate/volume/bitrate controls. `instructions` and `optimize_instructions` require the Instruct family. The provider enforces the instruction text's 1,600-token limit and Chinese/English language requirement; no character-count approximation replaces it. [Client event reference](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-client-events).

## Concurrent input and output

```rust,no_run
use std::sync::Arc;
use lingxi_llm_client::{
    RequestOptions,
    realtime::RealtimeTransport,
    qwen_tts_realtime::{
        QwenTtsRealtimeConfig, QwenTtsRealtimeError, QwenTtsRealtimeEventKind,
        QwenTtsRealtimeLimits, QwenTtsRealtimeRegion, QwenTtsRealtimeRequest,
        QwenTtsRealtimeService, QwenTtsRealtimeVoice,
    },
};

async fn speak(
    transport: Arc<dyn RealtimeTransport>,
    options: &RequestOptions,
) -> Result<(), QwenTtsRealtimeError> {
    let config = QwenTtsRealtimeConfig::new(
        "qwen-intl", "billing-account-1", QwenTtsRealtimeRegion::Singapore,
        "workspace-id",
    );
    let service = QwenTtsRealtimeService::new(transport, config)?;
    let request = QwenTtsRealtimeRequest::new(
        "qwen3-tts-flash-realtime", QwenTtsRealtimeVoice::System("Cherry".into()),
    );
    let session = service.connect(options, &request, QwenTtsRealtimeLimits::default()).await?;
    let (mut input, mut events) = session.into_parts();
    // Borrow both halves so the input handle stays alive until output finishes.
    let write = async {
        input.append_text("Hello. ").await?;
        input.append_text("This text arrived in a second chunk.").await?;
        input.finish().await
    };
    let read = async {
        while let Some(event) = events.next().await? {
            match event.kind {
                QwenTtsRealtimeEventKind::AudioDelta { data, .. } => {
                    // Feed the configured audio format to the host's player.
                    println!("{} audio bytes", data.len());
                }
                QwenTtsRealtimeEventKind::ResponseDone { status, .. } => {
                    println!("response status: {status}");
                    // event.native retains the complete response and usage.
                }
                QwenTtsRealtimeEventKind::SessionFinished => println!("session complete"),
                _ => {}
            }
        }
        Ok::<(), QwenTtsRealtimeError>(())
    };
    futures::try_join!(write, read)?;
    Ok(())
}
```

`connect()` waits for `session.created`, sends `session.update`, and requires `session.updated` before exposing input. It validates the acknowledged model (including documented alias equivalence), session ID, voice, mode, format and sample rate. An explicitly returned `language_type` must also match; an omitted optional language echo is accepted. Setup events are retained in `metadata()` rather than repeated in the output stream. Every JSON client event carries a fresh UUID v4 generated before sending.

`append_text()` submits one native `input_text_buffer.append`. In `Commit` mode, call `commit()` to start synthesis; a known-empty buffer is rejected locally in both modes (before any append or after clear). Explicit commit is also valid in `ServerCommit` mode. After text is appended in server-commit mode, asynchronous provider segmentation can make its current buffer state uncertain, so the server remains authoritative for a later commit. `commit()` and `clear_buffer()` return after sending; further controls/text remain gated with `ControlPending` until their acknowledgements are read through `events.next()`. Keep reading while writing so those acknowledgements can progress.

`clear_buffer()` discards only uncommitted text. It neither cancels an active response nor discards its later audio events. This protocol does not invent an Omni-style `response.cancel`. `abort()` closes locally and does not claim that already accepted synthesis was unbilled. `ping()` delegates only to the transport's WebSocket Ping support, with a 125-byte bound and no automatic timer.

## Completion and failure

Audio deltas decode Base64 from `response.audio.delta`, preserving response/item IDs. `AudioDone` ends one audio part; `ResponseDone` retains the provider's status and native usage. Failed, incomplete or unfamiliar terminal statuses are not rewritten as success and close this client session. All event JSON remains in `event.native`; malformed frames and provider error JSON remain in the corresponding error. Debug output redacts native payloads and audio.

`finish()` stops accepting text and sends `session.finish`. `SessionFinished` is emitted only after the send succeeds, the matching session returns `session.finished`, all observed responses have finished, and the WebSocket reaches clean EOF. A provider marker followed by more data, a provider error, a transport error or premature EOF cannot become success. Canceling a send fails the session closed and wakes blocked readers; no frame is replayed. Canceling `events.next()` during terminal cleanup does not discard its pending event/error: the next call returns that result exactly once. EOF is retained while waiting for a pending finish send, so resuming a canceled read does not poll a terminated custom stream again. Dropping either half cancels a still-active session, so retain the input handle while draining the final output. [Server event reference](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-server-events).

Default local bounds are 4 MiB per frame, 512 MiB cumulative decoded audio, 64 KiB per text/instruction input, 16 extra setup events and 64 active responses. Each send has a 30-second deadline, and finish acknowledgement plus EOF has a 60-second deadline. The config's connection deadline defaults to 30 seconds. These are client resource bounds, not provider quotas. The host owns scheduling and playback; custom transports must obey the existing connection/drop contract. No live account validation is implied by mock tests.

The Qwen-Audio-TTS/CosyVoice `/api-ws/v1/inference` protocol, standalone ASR, voice creation and Qwen-Omni are separate capabilities.
