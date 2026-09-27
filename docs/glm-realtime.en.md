# GLM Realtime (voice WebSocket)

This module implements Zhipu's first-party GLM Realtime WebSocket contract for server-side use. It connects only to the mainland endpoint confirmed by the official guide, `wss://open.bigmodel.cn/api/paas/v4/realtime`; it does not infer an international Realtime endpoint from the regular HTTP API.

Source: the [official MetaGLM Realtime SDK protocol guide](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/GLM-Realtime-doc-for-llm.md). It documents WebSocket JSON events, Bearer-header authentication, `session.update`, Base64 WAV input, PCM/MP3 output, and the `response.audio.*`, `response.audio_transcript.*`, and `response.done` events.

## Connection and account scope

Each connection receives an explicit regional route, a non-secret account scope, and a fresh credential. The credential is sent only in that WebSocket handshake as `Authorization: Bearer …`. The host owns API keys and host-issued JWTs. `GlmRealtimeScope` records the profile name, account identity, region, and endpoint fingerprint; it does not store credentials.

```rust,no_run
use std::sync::Arc;
use lingxi_llm_client::{protocol::Secret, realtime::{
    GlmRealtimeRoute, GlmRealtimeScope, GlmRealtimeSession, GlmRealtimeConfig,
    RealtimeLimits, RealtimeTransport,
}};
async fn connect(
    transport: Arc<dyn RealtimeTransport>,
    credential: Secret<String>,
) -> Result<(), Box<dyn std::error::Error>> {
let route = GlmRealtimeRoute::mainland_china();
let scope = GlmRealtimeScope::new("glm-mainland", "billing-account-1", &route)?;
let (session, driver) = GlmRealtimeSession::connect(
    transport,
    route,
    scope,
    credential, // a per-connection Secret<String>
    GlmRealtimeConfig::default(),
    RealtimeLimits::default(),
).await?;
Ok(())
}
```

`connect` waits for `session.created`, sends the documented `session.update`, then waits for `session.updated` before returning. Events received during setup remain available to the caller. Run the returned `RealtimeDriver` concurrently with the session control and event receiver on the host executor.

The default session uses `input_audio_format: "wav"`, `output_audio_format: "pcm"`, `turn_detection.type: "server_vad"`, and explicit `beta_fields.chat_mode: "audio"`, `tts_source: "e2e"`, and `auto_search: false`. Instructions and PCM/MP3 output are typed configuration options.

## Input and events

- `RealtimeInput::Audio` accepts only `RealtimeAudioFormat::Encoded { mime_type: "audio/wav" }`. The bytes may be WAV fragments; the driver Base64-encodes them into `input_audio_buffer.append`. Raw PCM input is not supported by this adapter.
- With `server_vad`, the server detects speech, commits audio, and creates audio-turn responses. `CommitAudio` is rejected locally in this mode. A host can still use `ContinueResponse` to emit an explicit `response.create` after a function result.
- With `client_vad`, send audio, use `CommitAudio` to emit `input_audio_buffer.commit`, then use `ContinueResponse` to emit `response.create`.
- `RealtimeInput::Text` sends a GLM `conversation.item.create` containing `input_text`, followed by `response.create`.
- Configure host-executed function tools with `GlmRealtimeConfig.tools`. GLM's session wire shape is flattened (`type`, `name`, optional `description`, and JSON Schema `parameters`). `tool_choice` supports `auto`, `none`, `required`, or `{ "type": "function", "function": "tool_name" }`.
- `response.function_call_arguments.done` is exposed as `FunctionCallArgumentsDone` with the original native event. Its `call_id` remains optional because the published guide omits it from the event table/example. Submit a result only when the event supplies a usable `call_id`; the adapter never synthesizes correlation IDs. `ToolResult` and `ToolResults` send `conversation.item.create` function-output items with the caller's exact `call_id` and compact JSON output. They do not start a response; send `ContinueResponse` separately after the host has handled the tool call.
- `Interrupt` maps to `response.cancel`. Device capture, VAD decisions, playback, and audio queues remain host responsibilities.
- `response.audio.delta` output is decoded to bytes while response/item/index fields are retained. The PCM/MP3 label comes from the session configuration. The official contract does not specify a PCM sample rate, so the adapter does not guess one.
- GLM documents `response.audio_transcript.delta` and `response.audio_transcript.done` as text output; these map to `TextDelta` and `TextDone`. The provider says this transcript is generated independently and may differ from or be absent relative to the model output, so it is not authoritative text. `response.done` retains the complete native response, including provider-supplied usage.
- `error` is surfaced as `ProviderError` with its native JSON; the official guide says most event errors do not close the session. Unknown events are retained as `Native`.

The current public GLM Realtime guide does not confirm an international WebSocket endpoint, so this adapter has no international route. There is a source discrepancy worth preserving: the protocol guide's event table/example omits `call_id`, while the current [MetaGLM Python SDK model](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/python/rtclient/models.py) defines optional `call_id` on both function-arguments-done and function-output items. The guide's [function-output example](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/GLM-Realtime-doc-for-llm.md) omits the ID too, but the SDK schema permits it; its [sample handler](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/python/samples/message_handler.py) likewise does not demonstrate the ID. This adapter follows the SDK schema while retaining the event's native JSON, and local contract tests do not establish live provider acceptance. Binary WebSocket frames and automatic search-tool execution remain outside this adapter. A remote disconnect emits `ConnectionInterrupted` and ends the driver. The client does not reconnect or replay input automatically.

`RealtimeInput::ClearAudio` sends `input_audio_buffer.clear` to discard uncommitted input audio. Its acknowledgement remains a native provider event. It does not cancel output or close the session; `FinishSession` remains unsupported.

Contract: [client events](https://github.com/MetaGLM/glm-realtime-sdk/blob/main/GLM-Realtime-doc-for-llm.md).
