# OpenAI GPT-Live primary WebSocket

`providers::openai::live::OpenAiLiveSession` implements OpenAI's GPT-Live primary WebSocket protocol. GPT-Live is a separate API from OpenAI Realtime: it connects to `wss://api.openai.com/v1/live/sessions` without query parameters, sends `session.start` first, and waits for `session.started`. It does not use `/v1/realtime?model=...` or OpenAI Realtime event names.

依据：[GPT-Live WebSocket 指南](https://developers.openai.com/api/docs/guides/voice-websockets)、[Primary WebSocket API 参考](https://developers.openai.com/api/reference/resources/live/primary-websocket)、[会话管理](https://developers.openai.com/api/docs/guides/live-conversations)、[委派与工具](https://developers.openai.com/api/docs/guides/live-delegation)。

## 连接、账户与凭证

`OpenAiLiveRoute::new()` 固定使用官方主 WebSocket endpoint；route 不接受查询参数或自定义 endpoint。`OpenAiLiveScope` 将本地 profile、account scope 与 endpoint fingerprint 绑定。API key 作为 `Secret<String>` 逐次传入连接，并只发送在 `Authorization: Bearer …` header 中。可选的安全标识通过 `OpenAI-Safety-Identifier` 发送；应由应用生成稳定且不直接暴露身份的值。

下面的函数构造 session 控制句柄、事件流和 driver。Host 必须在自己的 async executor 上同时运行 `driver.run()` 和事件/命令处理任务。

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

`connect` returns only after it receives `session.started` with the required model field matching the requested model; startup errors, a missing model, or a different returned model fail the connection. Session configuration selects `gpt-live-1`, initial instructions, optional text history, one audio format for both directions, optional voice, and either client or Responses delegation.

## 启动历史与存储

`OpenAiLiveConfig.input` 可在 `session.start` 中种入先前的文本对话。每条 `OpenAiLiveHistoryMessage` 只包含一个文本片段，角色为 `Developer`、`User` 或 `Assistant`；前两者编码为 `input_text`，助手内容编码为 `output_text`。空历史会省略 `input` 字段。客户端在连接前限制为最多 128 条消息；文档规定的 8,192 个合并 token 上限由服务端执行，客户端不猜测 tokenizer。

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

`store` 默认为 `false`，只有显式设置 `OpenAiLiveConfig.store = true` 才会在 `session.start` 中启用服务端存储。这会让已完成会话可用于后续 GPT-Live fork；项目必须启用存储并允许相应数据策略，官方文档说明保存记录保留 30 天，Zero Data Retention 会将其视为关闭。当前接口不创建 fork 连接，也不管理保存记录。

## Audio and events

`OpenAiLiveAudioFormat` supports mono raw PCM16LE at 24 kHz (the default) or 16 kHz, plus G.711 μ-law or A-law at 8 kHz. The audio format is fixed for input and output for the session. Pass raw sample bytes to `append_audio`; do not include WAV/container headers. PCM16 chunks must contain complete two-byte samples. Capture, chunk sizing, pacing, and playback queues belong to the host.

GPT-Live consumes appended audio continuously and manages listen/speak turns itself. This adapter does not expose Realtime-style `commit` or `clear` input-buffer controls.

The adapter exposes input and output transcript deltas with their session-relative timestamps, output audio decoded from Base64 using the configured format, context-append acknowledgments, delegation-created metadata, cumulative usage snapshots, provider errors, and unknown native events. `ResponseEvent` retains the complete nested Responses event and its outer `delegation_id`; dispatch on the nested event's own `type`. Transcript deltas are fragments and do not mark complete turns. Primary WebSocket output-audio events do not provide audio timing or an audio-done event. Each `session.usage.updated` value is cumulative; treat it as a replacement snapshot, not a delta to sum.

## Caller-owned delegation

`OpenAiLiveDelegation::Client` leaves backend work to the application. `OpenAiLiveDelegation::Responses` configures a separate Responses model and the documented function and `web_search` tool kinds. `OpenAiLiveResponsesConfig` exposes a bounded subset: model, instructions, tools, tool choice, parallel tool calls, and maximum output tokens. The provider remains authoritative for model/tool availability. Live also documents service tier, reasoning, and text settings; those settings are not exposed by this slice.

The adapter never executes application functions. In Responses mode, `append_responses_text` and `function_output` each send one `response.item.create`. `continue_responses` is a separate `response.create` command; call it only after the host has collected and returned every required function result. In client mode, the host owns context and backend processing; use `append_instructions`, `append_thinking`, or `append_commentary` to add context to the Live conversation. For these commands, `delegation_id` is required by the wire schema but represented as `Option`: use `None` for general context or Responses delegation, and an existing client delegation ID in client mode. If a command entered the local queue just before the terminal event, the driver drops it after `session.closed` and emits `CommandDroppedAfterSessionClosed` with its event ID when present.

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
    // Read response.event envelopes and run any authorized function calls in
    // the host. After sending every required result, continue explicitly:
    control.function_output("call_1", "{\"status\":\"shipped\"}", Some("item_2".into()))?;
    control.continue_responses(Some("continue_1".into()))
}
```

The `session.update` command changes only the exposed sparse Responses settings. It cannot switch delegation modes or alter the Live model, audio format, voice, or initial instructions. The provider may reject unsupported model-specific values or settings.

## Session shutdown

Call `request_close` when the host decides to end the session, then keep consuming events until `SessionClosed` arrives with final session usage. Only after observing that event should the host call `close_after_session_closed` for an orderly WebSocket shutdown. A remote close after `session.closed` is treated as orderly; an EOF before it is a connection interruption. `abort` remains available when startup, finalization, or transport handling fails; it does not confirm final usage. Dropping or aborting the transport never retries or reconnects.

Tests use an injected fake transport and do not make live OpenAI API requests.
