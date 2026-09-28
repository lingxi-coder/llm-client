# Qwen standalone ASR Realtime

`qwen_asr_realtime` implements audio-to-text recognition using Qwen3-ASR's native WebSocket protocol. It shares only the injectable `RealtimeTransport` with other realtime services. The host drives concurrent input/output; the client does not capture audio, transcode, reconnect, resend audio, or refresh credentials.

## Regional scope and models

`QwenAsrRealtimeConfig` binds profile, non-secret account identity, region and workspace. Connections use the region's workspace endpoint:

| Region | Endpoint |
| --- | --- |
| Beijing | `wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=...` |
| Singapore | `wss://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=...` |

Credentials are supplied for each connection through `RequestOptions`. A supplied `account_scope` must match the configured service. The handshake sends `Authorization: Bearer ...` and `X-DashScope-WorkSpace`; data inspection is omitted by default and emits `X-DashScope-DataInspection: enable` only when explicitly configured. API keys must match the selected account/region. [Connection guide](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-interaction-process).

Both regions support `qwen3-asr-flash-realtime`, `qwen3-asr-flash-realtime-2026-02-10`, and `qwen3-asr-flash-realtime-2025-10-27`. The stable ID is documented as equivalent to the 2025-10-27 version; that alias equivalence is accepted in session acknowledgement. Unknown model IDs are rejected before connecting. [Model card](https://help.aliyun.com/en/model-studio/qwen3-asr-flash-realtime), [supported regions and models](https://help.aliyun.com/en/model-studio/real-time-speech-recognition-user-guide).

## Audio and session options

`QwenAsrRealtimeRequest` defaults to PCM, 16 kHz and server VAD (`threshold: 0.2`, `silence_duration_ms: 800`). Supported formats are `Pcm` and `Opus`; supported sample rates are 8 and 16 kHz. PCM input must be mono, signed 16-bit little-endian samples and each append must contain complete samples. The client does not parse or resample encoded Opus packets.

The VAD threshold must be finite and between -1 and 1; silence duration is 200–6000 ms. `Manual` sends `turn_detection: null`. Optional language hints use `QwenAsrLanguage`; `corpus_text` maps to `input_audio_transcription.corpus.text`. The provider enforces the documented 10,000-token context limit; the client's byte limit is only a resource bound. [Client events](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-client-events).

`connect()` consumes `session.created`, sends a UUID-v4 `session.update`, and waits for `session.updated`. Setup JSON remains available in `metadata()`. Model/session identity and audio format are checked; the documented `pcm16` server echo is accepted for a `pcm` request. Optional echoed sampling rate, VAD, language and corpus settings must not contradict the request. The official server-event page contains an unrelated GPT model in an example: this client requires the actual requested Qwen model rather than copying that example literally.

## Concurrent use

```rust,no_run
use std::sync::Arc;
use lingxi_llm_client::{
    RequestOptions,
    realtime::RealtimeTransport,
    providers::qwen::asr_realtime::{
        QwenAsrRealtimeConfig, QwenAsrRealtimeError, QwenAsrRealtimeEventKind,
        QwenAsrRealtimeLimits, QwenAsrRealtimeRegion, QwenAsrRealtimeRequest,
        QwenAsrRealtimeService,
    },
};

async fn recognize(
    transport: Arc<dyn RealtimeTransport>,
    options: &RequestOptions,
    pcm16_mono_16khz: &[u8],
) -> Result<(), QwenAsrRealtimeError> {
    let config = QwenAsrRealtimeConfig::new(
        "qwen-intl", "billing-account-1", QwenAsrRealtimeRegion::Singapore,
        "workspace-id",
    );
    let service = QwenAsrRealtimeService::new(transport, config)?;
    let request = QwenAsrRealtimeRequest::new("qwen3-asr-flash-realtime");
    let session = service.connect(options, &request, QwenAsrRealtimeLimits::default()).await?;
    let (mut input, mut events) = session.into_parts();
    let write = async {
        for chunk in pcm16_mono_16khz.chunks(3200) {
            input.append_audio(chunk).await?;
        }
        input.finish().await
    };
    let read = async {
        while let Some(event) = events.next().await? {
            match event.kind {
                QwenAsrRealtimeEventKind::PartialTranscript { text, stash, .. } => {
                    println!("current preview: {text}{stash}");
                }
                QwenAsrRealtimeEventKind::TranscriptCompleted { transcript, .. } => {
                    println!("final transcript: {transcript}");
                }
                QwenAsrRealtimeEventKind::TranscriptFailed { code, .. } => {
                    println!("recognition failed: {code:?}");
                }
                QwenAsrRealtimeEventKind::SessionFinished => println!("session complete"),
                _ => {}
            }
        }
        Ok::<(), QwenAsrRealtimeError>(())
    };
    // Both handles remain alive until the final event is drained.
    futures::try_join!(write, read)?;
    Ok(())
}
```

In VAD mode, send audio continuously and finish when input ends. Calling `commit()` in VAD mode fails before sending. In Manual mode, append an utterance then call `commit()`; an empty buffer is rejected. Continue reading the commit acknowledgement before another append/commit. A successful commit send may be followed immediately by `finish()` before its acknowledgement is read, matching the official flow. A manual session with uncommitted local audio must commit before finishing.

The referenced standalone ASR client/server contracts do not document `input_audio_buffer.clear`, so this API has no clear method and does not borrow one from Omni or TTS. `abort()` closes locally; `ping()` is an explicit transport-level WebSocket Ping, not a provider JSON event or automatic keepalive.

## Transcript and completion semantics

`PartialTranscript.text` is the currently confirmed prefix; `stash` is its revisable suffix. Replace the current item's preview with their concatenation. Neither field is an append-only delta, and an empty stash does not make the item final. `TranscriptCompleted.transcript` is the final recognition result. Item ID, content index, language, emotion and native payload remain available. `conversation.item.created` reporting item status `completed` is not transcription completion. This model's transcript API does not supply word timestamps; speech start/stop offsets are separate VAD events. [Server events](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-server-events), [recognition guide](https://help.aliyun.com/en/model-studio/real-time-speech-recognition-user-guide).

All event JSON and any provider usage/extensions are retained in `event.native`. Provider `error` events remain native error payloads. Item-specific failure is returned as `TranscriptFailed` and removes only that item from the outstanding recognition set. Other items and subsequent input continue; the client does not resend the failed audio. Generic provider errors, transport failures and malformed lifecycle events still terminate the session. `SessionFinished` means the session has drained, not that every item was successfully transcribed: callers must track failed items separately. Raw payloads and transcript text are redacted from Debug output.

ASR requires **client-initiated close after `session.finished`**. The receiver waits until the `session.finish` send is confirmed, initiates WebSocket close, and keeps draining inbound events concurrently. `SessionFinished` requires the provider marker, no unfinished observed items, confirmed local close and clean EOF. No-speech sessions may finish without any transcript. Late provider/transport errors and premature EOF cannot become success.

The close future is retained across canceled `events.next()` calls and holds no shared sink lock while the reader progresses. Pending terminal events/errors and observed EOF survive canceled reads. A canceled audio/control send fails the session closed and wakes blocked readers; there is no resend. Keep the input handle alive while draining completion, because dropping either half cancels an active session.

Default client bounds are 4 MiB/frame, 1 MiB/raw audio append, 512 MiB/session input audio, 64 KiB/text event or corpus, 16 MiB/cumulative transcript text, 16 extra setup events, and 64 active items. Defaults are 30 seconds per send and 60 seconds for final recognition/close; connection setup defaults to 30 seconds and is capped by `RequestOptions::total_timeout` if supplied. These are local bounds, not provider quotas. No live-account validation is implied by mock tests.
