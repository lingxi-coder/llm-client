# Gemini Interactions API

[中文](interactions.md)

Bind `client.provider::<GoogleClient>(profile)?` to an exact profile, then use `provider.interactions()`. Each operation takes `RequestOptions`; the client retains no credential.

`provider.interactions()` calls Google's [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview), separate from `generateContent` Chat. The built-in `gemini` profile configures the `/v1beta/interactions` route and `x-goog-api-key` authentication. Requests and responses retain the native Interactions API structure, including multimodal `input`, tool declarations, `steps`, status, and usage.

`InteractionRequest::model()` and `::agent()` still accept string input. `InteractionInput` also encodes one native `Content` block, an array of `Content`, or an array of `Step` objects. Content blocks support text, images, audio, documents, and video; media can be provided as base64 `data` or a file `uri`:

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::google::interactions::{
    InteractionContent, InteractionInput, InteractionRequest,
};

async fn describe_media(client: &LlmClient, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::google::GoogleClient>("gemini")?;
    let input = InteractionInput::content([
        InteractionContent::text("Summarize this recording and its cover image."),
        InteractionContent::image_data("BASE64_IMAGE", "image/png"),
        InteractionContent::audio_uri(
            "https://generativelanguage.googleapis.com/files/audio-id",
            Some("audio/mp3".into()),
        ),
    ]);
    let request = InteractionRequest::model("gemini-3.8-flash", input);
    let interaction = provider.interactions().create(&request, options).await;
    // Inspect the native steps and usage on the returned interaction.
    let _ = interaction;
    Ok(())
}
```

Declare client-executed functions with `InteractionTool::function`. Its `parameters` argument is the JSON Schema sent to Google. `generation_config` is sent as provided, so tool selection uses Google's `generation_config.tool_choice` shape:

```rust,no_run
use lingxi_llm_client::providers::google::interactions::{InteractionRequest, InteractionTool};
use serde_json::json;

let request = InteractionRequest::model("gemini-3.8-flash", "Weather in Paris?")
    .with_tool(InteractionTool::function(
        "get_weather",
        "Gets the weather for a city.",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    ))
    .with_generation_config(json!({
        "tool_choice": {
            "allowed_tools": {"mode": "any", "tools": ["get_weather"]}
        }
    }));
```

Tool execution stays with the caller. The response's `native["steps"]` preserves Google's `function_call` and other step objects. To return a function result, continue the stored interaction and provide a native `function_result` step, using the exact call ID returned by the model:

```rust,no_run
use lingxi_llm_client::providers::google::interactions::{InteractionInput, InteractionRequest, InteractionResult};
use serde_json::json;

fn continue_with_result(prior: &InteractionResult, call_id: &str) -> InteractionRequest {
    let mut next = InteractionRequest::model(
        "gemini-3.8-flash",
        InteractionInput::steps([json!({
            "type": "function_result",
            "name": "get_weather",
            "call_id": call_id,
            "result": [{"type": "text", "text": "{\"weather\":\"sunny\"}"}]
        })]),
    );
    next.previous = prior.reference.clone();
    // Include the function declaration again when the model may call it again.
    next
}
```

Google Search, URL Context, and Code Execution have convenience constructors. Other provider-native tool declarations can be passed through `InteractionTool::from_value`. Google currently states that Gemini 3 Interactions do not support Remote MCP, so an `mcp_server` tool with a Gemini 3 model is rejected before sending. The client encodes requests but does not execute client tools or reinterpret Google's output.

The caller supplies a request-scoped credential and non-secret `account_scope`. `InteractionRef` is bound to the provider, profile, route, and account; a mismatched continuation or retrieval fails before HTTP. A `store=false` result has no retrievable reference, and background work must be stored. `InteractionRequest::agent()` defaults to `background=true`. Submission and background execution can have side effects, so a transport interruption reports an unknown outcome without automatic resubmission or failover. Encoded request bodies are limited to 100 MiB; for larger media, upload through Google's Files API and pass the resulting file URI.

`create_stream()` uses the same input and tool encoder and sends `stream=true` to the same endpoint. It yields Google's native [streaming events](https://ai.google.dev/gemini-api/docs/streaming) with `event_type`, JSON, and `event_id`. The stream is complete only after `[DONE]`; premature EOF reports an interruption with any account-bound interaction reference already received. Save `reference()` and the latest `last_event_id()`, then call `resume_stream()` to continue. That call makes one `GET /interactions/{id}?stream=true&last_event_id=...` request and does not retry automatically.

Use `cancel()` to cancel a background interaction that is still running; `delete()` removes the server-side record. Both operations require an account-bound `InteractionRef`. Google does not document how long event cursors remain valid; expired or invalid cursors are returned as provider errors. Tests use a local mock and make no live Google account calls.
