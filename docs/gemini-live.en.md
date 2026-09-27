# Gemini Live WebSocket sessions

`realtime::GeminiLiveSession` uses the existing `RealtimeTransport` contract to establish a Google Gemini Live `BidiGenerateContent` session. The host supplies authentication in the endpoint or headers, runs the returned `RealtimeDriver`, and owns tool authorization, audio devices, and playback state.

The built-in `RustlsWebSocketTransport` requires the `realtime-websocket` dependency feature. A host can also inject its own `RealtimeTransport`.

Google's raw WebSocket endpoint is `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`. A standard API key can be passed as the `?key=...` query parameter. Google documents short-lived tokens in either the `access_token` query parameter or an `Authorization: Token ...` header. Keep these values out of logs.

```rust,ignore
use std::sync::Arc;
use lingxi_llm_client::realtime::{
    GeminiLiveConfig, GeminiLiveEvent, GeminiLiveSession, RealtimeConnectRequest,
    RealtimeInput, RealtimeLimits, RealtimeTransport, RustlsWebSocketTransport,
};

let transport: Arc<dyn RealtimeTransport> = Arc::new(RustlsWebSocketTransport);
let mut config = GeminiLiveConfig::new("gemini-3.8-live", "google-account-a");
config.system_instruction = Some("Answer briefly.".into());
config.enable_session_resumption = true;

let (session, driver) = GeminiLiveSession::connect(
    transport,
    RealtimeConnectRequest {
        endpoint: format!("{google_live_websocket_url}?key={api_key}"),
        headers: vec![],
        max_frame_bytes: 1024 * 1024,
    },
    config,
    RealtimeLimits::default(),
).await?;
let (control, mut events) = session.into_parts();
// Run `driver.run()` concurrently on the host executor.
control.send(RealtimeInput::Text("Hello".into()))?;
while let Some(event) = events.next().await {
    match event {
        GeminiLiveEvent::SessionResumptionUpdate(update) => {
            // Persist update.handle when it is Some.
        }
        GeminiLiveEvent::Realtime(event) => {
            // Consume normalized audio/text/tool events under host policy.
        }
    }
}
```

`connect` sends the one initial `setup` JSON message and waits for Google's `setupComplete` reply before returning the session. The setup message selects the model and carries generation settings, an optional text-only system instruction, tools, and realtime input settings. OpenAI Realtime events are not sent to this endpoint.

Text is sent as `realtimeInput.text`. The host must provide mono, little-endian PCM16 at 16 kHz; the adapter validates the declared PCM format and sample rate, then sends base64 with MIME type `audio/pcm;rate=16000`. Split microphone audio into small chunks. With Google's default automatic activity detection, `CommitAudio` sends `audioStreamEnd: true`. When `realtimeInputConfig.automaticActivityDetection.disabled` is true, text is wrapped in `activityStart`/`activityEnd`, `Interrupt` sends `activityStart`, and `CommitAudio` sends `activityEnd`. `Interrupt` returns `InvalidInput` while automatic detection is enabled because the API permits manual activity signals only when detection is disabled.

Output audio and text parts become `AudioDelta` and `TextDelta`. `turnComplete` becomes `TurnCompleted`, and `interrupted` becomes `Interrupted`. Function calls become `ToolCall` events. A `ToolResult` must use the call ID from that event and contain a JSON object; the codec retains the matching function name to send Google's `toolResponse.functionResponses` message. The host authorizes and executes tool calls.

Set `enable_session_resumption = true` to receive resumption updates. A server `sessionResumptionUpdate` becomes a typed `GeminiLiveResumeUpdate`; it contains a handle only when Google marks the state resumable and supplies a non-empty `newHandle`. The handle can be serialized for host-managed persistence, while `Debug` redacts the provider token. It is bound to the model, endpoint origin/path, and the host-supplied non-secret `credential_scope`. Query parameters are omitted from that stored scope, so API keys and short-lived tokens are not persisted there. Use a stable credential scope per Google account and reuse a handle only with the same model, endpoint, and account scope. If query parameters also carry routing semantics, distinguish those configurations with `credential_scope`. A mismatched handle is rejected before connecting.

Google documents that resumption handles expire two hours after the session terminates, and that some states, including active generation and function calls, cannot resume. The client does not retry, reconnect, replay inputs, or resume automatically. The host decides whether and when to open another session with the latest resumable handle. Other server messages, including `goAway` and tool cancellation, remain available as native `ProviderEvent` data.

Enabling resumption stores conversation state on Google's side; Google's zero-data-retention guidance says state associated with a generated session handle can be retained for up to 24 hours. Do not enable it for conversations that must not be retained.

`RealtimeInput::Image` sends a non-empty JPEG or PNG frame as `realtimeInput.video`; capture, encoding and pacing remain with the host. Input bytes and the encoded frame obey the session limits.

`RealtimeInput::ToolResults` sends a batch in one `toolResponse.functionResponses` message. Every result must match a tracked call ID, use a JSON object response, and have a unique ID within the batch; cancellation removes the matching tracked call. Validation completes before any frame is sent. Gemini decides when to continue, so `ContinueResponse` remains unsupported.

New functionality awaits the unified test phase; no live Google calls were made.

References: [Gemini Live WebSocket quickstart](https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket), [WebSockets API reference](https://ai.google.dev/api/live), [session management](https://ai.google.dev/gemini-api/docs/live-api/session-management), [zero-data-retention guidance](https://ai.google.dev/gemini-api/docs/zdr), and [Live transcription audio format](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe).

Once a function response passes frame limits and enters the outbound queue, its call-name lookup is released. Queue-full/closed and invalid-frame failures keep the lookup for a later attempt. Queue acceptance does not acknowledge provider delivery; transport errors still end the driver without automatic replay.
