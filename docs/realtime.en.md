# Realtime sessions

`lingxi-llm-client::realtime` defines a provider-neutral session contract for
low-latency, bidirectional model APIs. The host runs the driver and owns
credentials, tools, audio capture, playback, permissions, and conversation
state. The network backend is optional: enable the `realtime-websocket`
feature to use `HttpTransport`. It requires a Tokio runtime and a
`wss://` endpoint, and never reconnects a live session automatically.

```rust,ignore
use std::sync::Arc;
use lingxi_llm_client::realtime::{
    OpenAiRealtimeCodec, RealtimeConnectRequest, RealtimeInput, RealtimeLimits,
    RealtimeSession, HttpTransport,
};

let transport = HttpTransport::new()?;
let (session, driver) = RealtimeSession::connect(
    &transport,
    RealtimeConnectRequest {
        endpoint: realtime_endpoint,
        headers: authenticated_headers,
        max_frame_bytes: 1024 * 1024,
    },
    Arc::new(OpenAiRealtimeCodec::default()),
    RealtimeLimits::default(),
).await?;
let (control, mut events) = session.into_parts();
// Run `driver.run()` concurrently on the host's executor.
control.send(RealtimeInput::Text("Hello".into()))?;
while let Some(event) = events.next().await {
    // Handle normalized events and dispatch ToolCall through host policy.
}
```

Enable `features = ["realtime-websocket"]` in the dependency declaration to
import `HttpTransport`.

The built-in transport uses tokio-tungstenite, Rustls, and bundled Mozilla root
certificates for the WebSocket handshake and TLS. It sends the host-provided
headers as supplied, handles ping/pong in the WebSocket layer, and returns a
writer plus a stream of complete data messages. It does not retry or reconnect
implicitly, and it does not log endpoint or header credentials. Dropping the
driver releases both transport halves. Hosts can still inject their own
`RealtimeTransport` when the optional feature is disabled.

Outbound input uses a bounded queue. `RealtimeControl::send` and
`RealtimeControl::interrupt` return `QueueFull` immediately when the queue is
full, so a microphone loop can drop or coalesce old chunks instead of building
unbounded latency. Every encoded and received frame is checked against
`max_frame_bytes`. The event queue is bounded too; if the host stops consuming
events, the driver pauses reads and applies backpressure to the transport.

Calling `close` queues a close after earlier outbound messages and waits for the
transport's close operation. Dropping every control handle lets the driver
drain accepted messages, send a normal close, and publish `Closed`. Dropping the
driver itself aborts the connection by dropping the injected transport halves;
Rust `Drop` cannot guarantee an asynchronous close handshake.

`RealtimeControl::abort(close)` cancels pending sends and discards accepted
outbound messages immediately. It preempts a blocked transport send or a full
event queue and drops both connection halves without flushing a graceful
WebSocket close. A `Closed` event is best effort when aborting; the event stream
still terminates even when that event cannot fit. Injected sinks may implement
the synchronous `RealtimeSink::abort` cleanup hook, which must never drain
network writes.

All initialization frames are generated and validated before connecting. An invalid configuration or oversized initialization frame prevents the transport handshake and any partial initialization writes.

A remote-initiated close is reported as a connection interruption. `Closed`
means that the local caller closed the session or dropped all control handles.

## Native Agent conversation

`connect_audio_conversation(snapshot, route, config, tools, history, credential,
transport, limits)` builds endpoint, authentication and provider tool/config
frames for the exact `AudioRoute`. `AudioRealtimeConfig` selects an independent
audio model and optional voice/instructions. Catalog defaults do not use the
chat model. The returned `ConnectedAudioConversation` carries bounded control,
events, driver, resolved model and input/output formats. It currently admits
OpenAI API Realtime and Gemini Developer Live; unsupported profiles, audio
models, and hosted/deferred tools fail before connection.

`ImportHistory` preserves message roles, function-call IDs and results without
requesting a response. OpenAI imports conversation items. Gemini uses the
[initial-history handshake](https://ai.google.dev/api/live#HistoryConfig):
`historyConfig.initialHistoryInClientContent` prevents the final initial
`clientContent.turnComplete` from generating a response. Import occurs once
before live input. System history uses Gemini's dedicated system instruction.

Both connectors use host-controlled input boundaries: send audio chunks,
`CommitAudio`, then `ContinueResponse`. Gemini starts/ends manual activity in
its adapter and resumes automatically after function results, so its explicit
continuation is a local no-op. Generic JSON tool results that are not objects
are wrapped as `{ "result": value }` for Gemini's object-only native response.
The SDK never executes a function.

`Transcript` distinguishes input/output and `Delta`/`Replace`, and final text
is a complete item snapshot. Provider item/response IDs are retained; Gemini
uses local stream item/turn IDs because its protocol supplies no such IDs.
`ToolCancelled`, `Usage` and `SessionResumption` preserve their native envelopes
alongside normalized events. Gemini usage has no response ID or guaranteed
billable delta semantics, so its `Usage.turn_id` is `None`; callers must not
infer cost by subtracting prompt-context counters. OpenAI usage is attributed
to the provider response ID. Every recognized native event is also retained.

`control.capabilities()` reports the actual configured contract. xAI and GLM
provide normalized generic streams through `into_realtime_parts`, but do not
advertise Agent conversation until history import is implemented. Truncation
and resumption have separate flags: OpenAI truncates audio but does not return
an exact surviving transcript prefix; Gemini interruption does not implement
audio truncation. Scoped provider resume handles remain opt-in.

## OpenAI Realtime codec

`OpenAiRealtimeCodec` emits a `session.update` as the first client event after
the transport connects. The model and authentication come from the endpoint
and headers supplied by the host. `OpenAiRealtimeConfig.tools` declares
session-level function tools, and `tool_choice` supports `Auto`, `None`,
`Required`, or a named configured `Function`. Function declarations expose
only the Realtime-documented `type`, `name`, optional `description`, and JSON
Schema `parameters`. Duplicate names, non-object parameters, or a choice that
names an undeclared function are rejected before connecting.

Text input becomes a user conversation item and `response.create`; audio
becomes `input_audio_buffer.append`; manual audio finalization becomes
`input_audio_buffer.commit`; `ClearAudio` sends `input_audio_buffer.clear` to
discard uncommitted input audio, which the server acknowledges with
`input_audio_buffer.cleared`. An empty commit is a provider error, and commit
does not create a model response. Interruption becomes `response.cancel`.
OpenAI Realtime WebSocket has no `FinishSession` client event; the host closes
the transport when done. Image
input becomes an `input_image` content part with a Base64 data URL. The codec
uses only bytes supplied by the host: it does not make a separate file upload
request or fetch remote URLs. OpenAI currently documents image input for
`gpt-realtime-2` and `gpt-realtime`; other model availability is decided by
the service. The image item is inserted without automatically creating a
response; the host can send `ContinueResponse` (`response.create`) when ready.
`OpenAiRealtimeConfig.voice` encodes to `session.audio.output.voice` and
optionally selects a named built-in voice (a string) or a project-available
custom voice ID (`{ id }`). Voice names are not hardcoded, and OpenAI validates
availability for the project. A single
`ToolResult` and `ToolResults` insert function-call output items without
starting another response. The host submits every completed result before
sending `ContinueResponse`
(`response.create`) when it is ready for inference to resume. This places all
tool outputs in the conversation before the next model response. The client
does not execute tools or decide when to continue.

The default session uses mono, little-endian PCM16 at 24 kHz and audio output.
The OpenAI Realtime reference describes this PCM rate for Realtime sessions; the input event's
[`RealtimeAudioFormat`] must match the codec setting. The host can instead use
G.711 μ-law or A-law. Text-only output is configurable. The codec normalizes
audio and text deltas, completed responses, function calls, provider errors,
and speech-start events. Unknown provider events retain their native JSON.
Encoded image content is still subject to `max_frame_bytes`.

Conversation items also support explicit `RetrieveItem`, `DeleteItem`, and
`TruncateAudio` operations, encoded as `conversation.item.retrieve`,
`conversation.item.delete`, and `conversation.item.truncate`. Truncation is
only for assistant audio messages and OpenAI requires `content_index` to be
`0`. The host supplies `audio_end_ms` from actual playback progress; the
provider rejects a value beyond the original audio duration. On success, the
provider also removes transcript text for audio the user has not heard.
Retrieve, delete, and truncate do not create a model response.

Tool calls are delivered for the host to authorize and execute; this module
never executes tools. Each `ToolResults` group must contain 1–128 results and
cannot contain empty or duplicate `call_id` values. The 128-result bound is a
client queue limit, not a provider limit. `response.cancel` stops generation.
If the host is playing audio, it must stop local playback and reconcile output
that the user has not heard. OpenAI documents `output_audio_buffer.clear` for
WebRTC/SIP, not WebSocket.

## Gemini Live protocol boundary

Gemini Live has its own bidirectional `BidiGenerateContent` WebSocket protocol.
`GeminiLiveSession` sends the first `setup` JSON message and waits for
`setupComplete` before returning the session to its host. Later client messages
use `realtimeInput` or `toolResponse`; server messages include `serverContent`,
`toolCall`, interruption, and resumption events. Never send OpenAI event JSON
to a Gemini endpoint. See [`gemini-live.en.md`](gemini-live.en.md) for details.

References: [OpenAI Realtime image input](https://developers.openai.com/api/docs/guides/realtime-conversations#image-inputs),
[OpenAI Realtime custom voices](https://developers.openai.com/api/docs/guides/custom-voices),
[OpenAI Realtime function tools](https://developers.openai.com/api/docs/guides/realtime-mcp#configure-a-function-tool),
[OpenAI Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events),
[OpenAI Realtime server events](https://developers.openai.com/api/reference/resources/realtime/server-events),
[Gemini Live WebSockets API](https://ai.google.dev/api/live), and
[Gemini Live WebSocket quickstart](https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket).

The built-in WebSocket backend does not resume a disconnected session. This
foundation also does not provide live-account acceptance, media
capture/playback, video frames, or provider-neutral tool execution. The codec
is a protocol adapter, not a claim of live provider availability.

For transcription-only xAI sessions at `/v1/stt`, use [XaiSttSession](xai-stt.en.md), which handles raw binary input and per-channel completion.

For bidirectional xAI text-to-audio streaming at `/v1/tts`, see [Streaming TTS](xai-streaming-tts.en.md), including multiple utterances, cancellation, and replacement-map updates.
