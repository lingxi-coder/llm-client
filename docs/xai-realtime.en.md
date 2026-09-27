# xAI Speech-to-Speech Realtime

`realtime::XaiRealtimeSession` adapts xAI's native `/v1/realtime` WebSocket protocol to the existing `RealtimeTransport` and bounded realtime queues. The host supplies authentication through the connection request and runs the returned `RealtimeDriver` concurrently with its event loop. JSON/base64 is the default audio transport; input and output can independently use raw binary WebSocket frames.

The built-in `RustlsWebSocketTransport` is available with the `realtime-websocket` feature. Direct sessions use `wss://api.x.ai/v1/realtime?model=grok-voice-latest`; xAI also documents `grok-voice-think-fast-2.0`. Authentication uses an `Authorization: Bearer ...` header. For client-side connections, xAI documents short-lived client secrets through the same bearer header or the `Sec-WebSocket-Protocol: xai-client-secret.<token>` value. Keep API keys and client secrets out of logs.

```rust,ignore
use std::sync::Arc;
use lingxi_llm_client::realtime::{
    RealtimeConnectRequest, RealtimeInput, RealtimeLimits, RealtimeTransport,
    RustlsWebSocketTransport, XaiRealtimeConfig, XaiRealtimeSession,
};

let transport: Arc<dyn RealtimeTransport> = Arc::new(RustlsWebSocketTransport);
let request = RealtimeConnectRequest {
    endpoint: "wss://api.x.ai/v1/realtime?model=grok-voice-latest".into(),
    headers: vec![("Authorization".into(), format!("Bearer {xai_token}"))],
    max_frame_bytes: 1024 * 1024,
};
let (session, driver) = XaiRealtimeSession::connect(
    transport,
    request,
    XaiRealtimeConfig::default(),
    RealtimeLimits::default(),
).await?;
let (control, mut events) = session.into_parts();
// Run `driver.run()` concurrently on the host executor.
control.send(RealtimeInput::Text("Hello".into()))?;
while let Some(event) = events.next().await {
    // Handle typed xAI session, text, audio, transcript, and response events.
}
```

The server first emits `session.created` and `conversation.created`. The adapter sends one `session.update` after `session.created`, then waits for `session.updated` before returning the session. These initial native events are replayed to the returned event stream in order. `XaiRealtimeEvent` types common session lifecycle, speech activity, transcription, audio/text output, response, function-argument, MCP, and error events. Less common or future event types retain their original JSON in `XaiRealtimeEvent::Native`; core close and connection events appear under `XaiRealtimeEvent::Realtime`.

Text input sends `conversation.item.create` with an `input_text` part and then `response.create`. By default, audio input uses base64 JSON `input_audio_buffer.append`. The default format is little-endian PCM16 at 24 kHz; the adapter also supports xAI's documented PCM rates, G.711 μ-law/A-law, and 24 kHz mono Opus packets. Input and output formats and JSON/binary transports can be configured independently. Binary input is sent as `RealtimeFrame::Binary`; binary output is returned as `XaiRealtimeEvent::AudioBinaryDelta`. Binary audio events have no JSON response/item IDs. JSON audio deltas continue to appear as `AudioDelta` with their native event and indices.

`XaiRealtimeConfig` also supports the documented reasoning effort (`high`/`none`), server VAD threshold (0.1–0.9), silence and prefix padding durations (each at most 10,000 ms), idle timeout, output speech speed (0.7–1.5), pronunciation replacement map, and client-side function tool declarations. The caller supplies each tool's JSON Schema; authorization and execution remain with the host. VAD options are valid only with server VAD. Invalid bounds are rejected before connecting.

Set `input_transcription = true` to request `grok-transcribe` input transcription. You can also set the BCP-47 `input_transcription_language_hint` and up to 100 `input_transcription_keyterms`, each at most 50 characters. Hints and keyterms require transcription to be enabled. xAI emits `conversation.item.input_audio_transcription.updated` as a cumulative transcript that may revise prior updates, followed by the completed transcript event.

The default `XaiTurnDetection::ServerVad` lets xAI detect end of speech, so audio is sent as append events and `CommitAudio` is rejected. Set `XaiTurnDetection::Manual` to configure `turn_detection.type` as `null`; then `CommitAudio` sends `input_audio_buffer.commit` and `Interrupt` sends `response.cancel`. `ClearAudio` or `XaiRealtimeCommand::ClearAudioBuffer` sends `input_audio_buffer.clear` to discard uncommitted input audio. Text input requests its response explicitly in either mode.

`XaiRealtimeControl::send_command` also supports xAI's `ForceMessage` extension (synthesize fixed text without a model response) and `CreateResponse` with a one-response instruction override. Provider commands and generic `RealtimeInput` share the same bounded outbound queue and frame-size limit. xAI documents delete and truncate conversation-item events but its Realtime reference does not specify their request fields; this adapter returns `InvalidInput` for those generic inputs rather than guessing a payload. xAI explicitly lists `conversation.item.retrieve` as unsupported.

Each `XaiRealtimeEvent` preserves the native event JSON in its `native` field. For example, `AudioDelta` exposes decoded bytes, response/item identifiers, content indices, and the configured output format; `InputTranscriptionUpdated` exposes xAI's cumulative transcript, which may revise earlier text. Audio-device access, playback, tool authorization, and conversation policy remain host responsibilities.

### Returning custom function results

Custom functions are executed by the host. After receiving `FunctionCallArgumentsDone`, send its provider-issued `call_id` and the result value. `ToolResult` submits one output item; `ToolResults` submits a batch of outputs in order as one bounded queue entry. Both serialize each result as JSON text in the documented `output` field. They do not request another response. Send `ContinueResponse` only after every function call in the turn has a result and the host is ready for xAI to speak again:

```rust,ignore
use lingxi_llm_client::realtime::{RealtimeInput, RealtimeToolResult};

control.send(RealtimeInput::ToolResults {
    results: vec![
        RealtimeToolResult { call_id: first_call_id, output: first_result },
        RealtimeToolResult { call_id: second_call_id, output: second_result },
    ],
})?;

// The host can wait until playback of the current response finishes.
control.send(RealtimeInput::ContinueResponse)?;
```

This separation follows xAI's parallel-tool rule: send every `function_call_output` before the single `response.create`. The batch must contain 1–128 unique, non-empty call IDs; 128 is this client's local queue guard, not an xAI limit. The configured frame-size limit also bounds encoded results. Invalid batches fail before queueing. If transport fails while frames are being sent, the client does not replay them automatically; the host must treat delivery as uncertain.

When `resumption_enabled` is set, the host must provide a stable, non-secret `credential_scope`. Save `conversation.id` from `ConversationCreated`, then create a scoped reference with `XaiRealtimeResumeRef::import(id, model, endpoint, credential_scope)` and pass it as `resume_ref` when reconnecting. Before connection, the adapter checks the reference's model, normalized endpoint route, and account scope, then writes `model` and `conversation_id` into the WebSocket URL. The reference's `Debug` omits the conversation ID and account scope. The server replays history, but the client does not reconnect automatically or replay inputs whose delivery was uncertain. The provider drops cached history after 30 minutes of inactivity. The host owns connection lifecycle and decides what to do with native function-call events. Regression coverage uses fake transports; it does not call a real xAI account or make paid requests.

References: [xAI Voice WebSocket API reference](https://docs.x.ai/developers/rest-api-reference/inference/voice), [Speech-to-Speech guide](https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech), and [xAI Voice overview](https://docs.x.ai/developers/model-capabilities/audio/voice).
