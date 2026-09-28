# Qwen LiveTranslate Realtime (WebSocket)

The Qwen LiveTranslate adapter in `realtime` implements Alibaba Cloud Model Studio's real-time speech/video translation WebSocket protocol. It has dedicated configuration and event types separate from Qwen-Omni Realtime: Qwen 3.5 incremental text carries confirmed `text` and tentative `stash`, while Qwen 3.8 uses append-only `delta` events. It uses Beijing and Singapore workspace endpoints and supports the Qwen 3.5 stable alias, its `2026-05-19` snapshot, and the Qwen 3.8 stable model ID listed in Alibaba's current guide.

Sources: Alibaba's [Qwen LiveTranslate model guide](https://help.aliyun.com/en/model-studio/qwen3-5-livetranslate-flash-realtime), [client events](https://help.aliyun.com/en/model-studio/live-translator-client-events), and [server events](https://help.aliyun.com/en/model-studio/live-translator-server-events).

## Route, account, and credentials

`QwenLiveTranslateRoute::new` takes a region and one workspace ID and constructs the regional WebSocket endpoint. Connections use the documented `model` query parameter and `Authorization: Bearer …` handshake header. `QwenLiveTranslateScope` binds the profile, account, region, workspace, and endpoint fingerprint. The API key is passed per connection as `Secret<String>` and is not stored in the scope.

```rust,no_run
use lingxi_llm_client::providers::qwen::live_translate::{Qwen35LiveTranslateConfig, QwenLiveTranslateConfig, QwenLiveTranslateRegion, QwenLiveTranslateRoute, QwenLiveTranslateScope, QwenLiveTranslateSession};
use std::sync::Arc;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{RealtimeDriver, RealtimeError, RealtimeLimits, RealtimeTransport},
};

async fn connect_live_translate(
    transport: Arc<dyn RealtimeTransport>,
    workspace_id: String,
    api_key: Secret<String>,
) -> Result<(QwenLiveTranslateSession, RealtimeDriver), RealtimeError> {
    let route = QwenLiveTranslateRoute::new(
        QwenLiveTranslateRegion::Beijing,
        workspace_id,
    )?;
    let scope = QwenLiveTranslateScope::new("qwen-live", "account-1", &route)?;
    let config = QwenLiveTranslateConfig::qwen35(
        Qwen35LiveTranslateConfig::default(),
    );
    QwenLiveTranslateSession::connect(
        transport,
        route,
        scope,
        api_key,
        config,
        RealtimeLimits::default(),
    )
    .await
}
```

Setup waits for `session.created`, sends `session.update`, and returns only after `session.updated`. The config enum keeps the two wire schemas separate: `Qwen35LiveTranslateConfig` encodes flat `modalities`, sample rate, PCM/Opus input, and optional `turn_detection: null`; `Qwen38LiveTranslateConfig` encodes `output_modalities` and nested `audio.input`/`audio.output`. Both configurations support text or text plus audio, not audio-only. Qwen 3.5 accepts 8/16 kHz mono PCM16 or Opus; Qwen 3.8 uses 16 kHz mono PCM16. Output audio is reported as the documented default: PCM16 at 24 kHz.

## Input and controls

- Audio uses `RealtimeInput::Audio` and becomes Base64 `input_audio_buffer.append`. PCM16 samples must be complete; Qwen 3.5 Opus sessions require `Encoded { mime_type: "audio/opus" }`. Capture and chunking belong to the host.
- An image frame uses `RealtimeInput::Image` and accepts JPG/JPEG. Alibaba's guide gives a 500 KB maximum before Base64; the client conservatively enforces 500,000 bytes. At least one audio append must be successfully sent before an image. The model guide limits image input to two frames per second; the host owns frame extraction and pacing.
- `clear_audio()` / `RealtimeInput::ClearAudio` sends `input_audio_buffer.clear` to discard uncommitted audio.
- Qwen 3.5 manual mode uses `commit_audio()` / `RealtimeInput::CommitAudio` to submit the current audio. Commit is rejected locally in server VAD mode and for Qwen 3.8. The Qwen 3.8 config exposes only the documented speaker-detection and server-VAD modes.
- `finish()` / `RealtimeInput::FinishSession` sends `session.finish` and does not close the connection automatically. To retain final source-transcription and translation results, keep reading events until `QwenLiveTranslateEvent::SessionFinished`, then explicitly call `close_after_finished()`. `close()` remains available for a host-requested abort when a session fails or stalls.

Ordinary text, function-tool results, continue-response, and interruption are rejected because they are not part of this LiveTranslate client-event protocol.

## Configuration and events

Qwen 3.5 can configure target/source language, source transcription, 8/16 kHz input, VAD threshold and silence duration, voice, voice-cloning mode, and same-language skip options (effective only for target `zh` or `en`). Terminology mappings have a local configuration bound of 1,000 entries, matching Alibaba's recommendation; this is not a claimed server-side maximum. Enabling source transcription selects the fixed `qwen3-asr-flash-realtime` model. Qwen 3.8 uses its own nested turn-detection schema; source transcription is always returned, and its config omits Qwen 3.5's same-language skip and voice-cloning fields. Alibaba's service remains authoritative for supported languages and voices.

The event enum preserves model-specific semantics:

- Qwen 3.5 reports partial output as `TextProgress35 { text, stash }` or `AudioTranscriptProgress35 { text, stash }`; `TextDone` / `AudioTranscriptDone` carry the complete final text.
- Qwen 3.8 reports `TextDelta38 { delta }` or `AudioTranscriptDelta38 { delta }`. Append each `delta` in arrival order; do not interpret it as Qwen 3.5's replaceable `text` / `stash` fields.
- Source transcription is `SourceTranscription35 { text, stash, .. }` in Qwen 3.5 and `SourceTranscriptionDelta38 { delta, .. }` in Qwen 3.8; completion and failure have separate typed events.
- `AudioDelta` contains decoded PCM16 24 kHz bytes; `ResponseDone` retains the complete native response; unmapped events remain available through `Native { event_type, native }`.
- `error` becomes `ProviderError`, and the host decides whether to end the session. The transport does not retry or reconnect automatically.

This adapter does not capture/play audio, extract or pace video frames, fetch remote image URLs, retry, or reconnect. It has not made live account requests; protocol tests use an injected fake transport.
