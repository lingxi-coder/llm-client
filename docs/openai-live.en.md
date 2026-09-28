# OpenAI GPT-Live primary WebSocket

`providers::openai::live::OpenAiLiveSession` implements OpenAI's GPT-Live primary WebSocket protocol. GPT-Live is a separate API from OpenAI Realtime: it connects to `wss://api.openai.com/v1/live/sessions` without query parameters, sends `session.start` first, and waits for `session.started`. It does not use `/v1/realtime?model=...` or OpenAI Realtime event names.

Sources: OpenAI's [GPT-Live WebSockets guide](https://developers.openai.com/api/docs/guides/voice-websockets), [Primary WebSocket reference](https://developers.openai.com/api/reference/resources/live/primary-websocket), [Managing GPT-Live sessions](https://developers.openai.com/api/docs/guides/live-conversations), and [Delegation and tools](https://developers.openai.com/api/docs/guides/live-delegation).

## Connection, account, and credentials

`OpenAiLiveRoute::new()` uses the fixed official primary WebSocket endpoint; the route does not accept query parameters or a custom endpoint. `OpenAiLiveScope` binds a local profile and account scope to the endpoint fingerprint. The API key is supplied per connection as `Secret<String>` and sent only in the `Authorization: Bearer …` header. An optional safety identifier is sent as `OpenAI-Safety-Identifier`; applications should use a stable value that does not directly reveal a person's identity.

The function below constructs the session controls, event stream, and driver. The host must run `driver.run()` concurrently with its event and command handling on its own async executor.

```rust,no_run
use lingxi_llm_client::providers::openai::live::{OpenAiLiveConfig, OpenAiLiveControl, OpenAiLiveEvents, OpenAiLiveRoute, OpenAiLiveScope, OpenAiLiveSession, OpenAiLiveDriver};
use std::sync::Arc;
use lingxi_llm_client::{
    protocol::Secret,
    realtime::{RealtimeError, RealtimeLimits, RealtimeTransport},
};

async fn connect_gpt_live(
    transport: Arc<dyn RealtimeTransport>,
    api_key: Secret<String>,
) -> Result<(OpenAiLiveControl, OpenAiLiveEvents, OpenAiLiveDriver), RealtimeError> {
    let route = OpenAiLiveRoute::new();
    let scope = OpenAiLiveScope::new("openai-profile", "account-1", &route)?;
    let (session, driver) = OpenAiLiveSession::connect(
        transport,
        route,
        scope,
        api_key,
        Some("hashed-user-42".into()),
        OpenAiLiveConfig::default(),
        RealtimeLimits::default(),
    )
    .await?;
    let (control, events) = session.into_parts();
    Ok((control, events, driver))
}
```

`connect` returns only after `session.started` with the required model field matching the requested model; startup errors, a missing model, or a different returned model fail the connection. The session configuration selects `gpt-live-1`, initial instructions, optional text history, one audio format for both directions, an optional voice, and either client or Responses delegation.

## Startup history and storage

`OpenAiLiveConfig.input` seeds prior text conversation in `session.start`. Each `OpenAiLiveHistoryMessage` contains one text part and a `Developer`, `User`, or `Assistant` role. Developer and user text is encoded as `input_text`; assistant text is encoded as `output_text`. Empty history omits the `input` field. The client rejects more than 128 messages before connecting. OpenAI documents an 8,192-token combined limit; the provider enforces that token bound because this client does not guess at tokenization.

```rust,no_run
use lingxi_llm_client::providers::openai::live::{OpenAiLiveConfig, OpenAiLiveHistoryMessage, OpenAiLiveHistoryRole};

fn resume_from_text_history() -> OpenAiLiveConfig {
    OpenAiLiveConfig {
        input: vec![
            OpenAiLiveHistoryMessage::new(
                OpenAiLiveHistoryRole::User,
                "I need help with my recent order.",
            ),
            OpenAiLiveHistoryMessage::new(
                OpenAiLiveHistoryRole::Assistant,
                "What is the order number?",
            ),
        ],
        ..OpenAiLiveConfig::default()
    }
}
```

`store` defaults to `false`. Set `OpenAiLiveConfig.store = true` to include the opt-in provider-side storage setting in `session.start`. Completed stored sessions can be used for a later GPT-Live fork; the project must enable storage and have a data policy that permits persistence. OpenAI documents a 30-day retention period, and Zero Data Retention treats storage as disabled. This interface does not create fork connections or manage stored recordings.

## Audio and events

`OpenAiLiveAudioFormat` supports mono raw PCM16LE at 24 kHz (the default) or 16 kHz, plus G.711 μ-law or A-law at 8 kHz. The format is fixed for both input and output for the session. Pass raw sample bytes to `append_audio`; do not include WAV/container headers. PCM16 chunks must contain complete two-byte samples. Audio capture, chunk sizing, pacing, and playback queues remain host-owned.

GPT-Live consumes appended audio continuously and manages listen/speak turns itself. This adapter does not expose Realtime-style `commit` or `clear` input-buffer controls.

The adapter exposes input/output transcript deltas with session-relative timestamps, Base64-decoded output audio in the configured format, context-append acknowledgments, delegation metadata, cumulative usage snapshots, provider errors, and unknown native events. `ResponseEvent` preserves the full nested Responses event and its outer `delegation_id`; dispatch on the nested event's own `type`. Transcript deltas are fragments, not complete turns. Primary WebSocket output-audio events have no audio timing or audio-done event. Each `session.usage.updated` value is cumulative, so treat it as a replacement snapshot rather than a delta to add.

## Caller-owned delegation

`OpenAiLiveDelegation::Client` leaves backend work to the application. `OpenAiLiveDelegation::Responses` configures a separate Responses model and the documented `function` and `web_search` tool kinds. `OpenAiLiveResponsesConfig` exposes a bounded subset: model, instructions, tools, tool choice, parallel tool calls, and maximum output tokens. The provider remains authoritative for model and tool availability. Live also documents service tier, reasoning, and text settings; this slice does not expose them.

The adapter never executes application functions. In Responses mode, `append_responses_text` and `function_output` each send a single `response.item.create`. `continue_responses` is a separate `response.create` command; call it only after the host has collected and returned every required function result. In client mode, the host owns context and backend processing; use `append_instructions`, `append_thinking`, or `append_commentary` to add context to the Live conversation. The wire requires `delegation_id` on these commands, represented as `Option`: use `None` for general context or Responses delegation, and an existing client delegation ID in client mode. If a command entered the local queue just before the terminal event, the driver drops it after `session.closed` and emits `CommandDroppedAfterSessionClosed` with its event ID when present.

```rust,no_run
use lingxi_llm_client::providers::openai::live::{OpenAiLiveControl, OpenAiLiveDelegation, OpenAiLiveResponsesConfig, OpenAiLiveResponsesTool, OpenAiLiveToolChoice};
use lingxi_llm_client::realtime::{RealtimeError};
use serde_json::json;

fn configure_backend() -> Result<OpenAiLiveDelegation, RealtimeError> {
    let lookup = OpenAiLiveResponsesTool::function(
        "lookup_order",
        json!({
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": false
        }),
    )
    .with_description("Look up an order by ID.")?;
    let mut responses = OpenAiLiveResponsesConfig::new("gpt-6-luna");
    responses.tools = vec![lookup];
    responses.tool_choice = Some(OpenAiLiveToolChoice::Auto);
    Ok(OpenAiLiveDelegation::Responses(responses))
}

fn submit_backend_work(control: &OpenAiLiveControl) -> Result<(), RealtimeError> {
    control.append_responses_text("Where is order A-17?", Some("item_1".into()))?;
    // Read response.event envelopes and run authorized functions in the host.
    // After sending every required result, continue explicitly:
    control.function_output("call_1", "{\"status\":\"shipped\"}", Some("item_2".into()))?;
    control.continue_responses(Some("continue_1".into()))
}
```

`session.update` changes only the exposed sparse Responses settings. It cannot switch delegation modes or alter the Live model, audio format, voice, or initial instructions. The provider may reject unsupported model-specific values or settings.

## Session shutdown

Call `request_close` when the host decides to end the session, then keep consuming events until `SessionClosed` arrives with final session usage. Only after observing that event should the host call `close_after_session_closed` for an orderly WebSocket shutdown. A remote close after `session.closed` is treated as orderly; EOF before it is a connection interruption. `abort` remains available when startup, finalization, or transport handling fails; it does not confirm final usage. Dropping or aborting the transport never retries or reconnects.

Tests use an injected fake transport and make no live OpenAI API requests.
