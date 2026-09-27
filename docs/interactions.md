# Gemini Interactions API

[English](interactions.en.md)

`client.interactions()` 调用独立于 `generateContent` Chat 的 Google [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)。内置 `gemini` profile 配置 `/v1beta/interactions` 路由和 `x-goog-api-key`。请求与响应保持 Interactions API 原生结构，包括多模态 `input`、工具声明、`steps`、状态和用量。

`InteractionRequest::model()` 和 `::agent()` 仍接受字符串输入。`InteractionInput` 还可编码单个原生 `Content`、`Content` 数组或 `Step` 数组。内容块支持文本、图片、音频、文档和视频；媒体可通过 base64 `data` 或文件 `uri` 传递：

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::interactions::{
    InteractionContent, InteractionInput, InteractionRequest,
};

async fn describe_media(client: &LlmClient, options: &RequestOptions) {
    let input = InteractionInput::content([
        InteractionContent::text("Summarize this recording and its cover image."),
        InteractionContent::image_data("BASE64_IMAGE", "image/png"),
        InteractionContent::audio_uri(
            "https://generativelanguage.googleapis.com/files/audio-id",
            Some("audio/mp3".into()),
        ),
    ]);
    let request = InteractionRequest::model("gemini-3.8-flash", input);
    let interaction = client.interactions().create("gemini", &request, options).await;
    // Inspect the native steps and usage on the returned interaction.
    let _ = interaction;
}
```

可以用 `InteractionTool::function` 声明由客户端执行的函数。`parameters` 参数是发送给 Google 的 JSON Schema。`generation_config` 会按原样发送，因此工具选择要遵循 Google 的 `generation_config.tool_choice` 结构：

```rust,no_run
use lingxi_llm_client::interactions::{InteractionRequest, InteractionTool};
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

工具仍由调用方执行。响应中的 `native["steps"]` 会保留 Google 返回的 `function_call` 和其他 step 对象。要返回函数结果，可续接已存储的 interaction，并传入原生 `function_result` step，使用模型返回的原始 call ID：

```rust,no_run
use lingxi_llm_client::interactions::{InteractionInput, InteractionRequest, InteractionResult};
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

Google Search、URL Context 和 Code Execution 等内置工具提供了便捷构造方法。其他 Google 原生工具声明可以通过 `InteractionTool::from_value` 传入。Google 当前明确说明 Gemini 3 的 Interactions 不支持 Remote MCP；`mcp_server` 与 Gemini 3 模型组合会在发送前拒绝。客户端只负责编码请求，不会执行调用方的工具，也不会改写 Google 的输出。

调用方须提供请求级凭证和非敏感 `account_scope`。`InteractionRef` 绑定 provider、profile、路由和账户；跨账户/路由续接或查询在发请求前拒绝。`store=false` 的结果不生成可查询引用；后台任务必须存储。`InteractionRequest::agent()` 默认 `background=true`。创建/后台执行有副作用，因此传输超时或断开会返回结果未知错误，不自动重提或故障转移。编码后的请求体最多 100 MiB；更大的媒体应先使用 Google Files API，再传入文件 URI。

`create_stream()` 使用同一输入和工具编码，并将 `stream=true` 发送到相同端点。它逐个返回 Google [流式事件](https://ai.google.dev/gemini-api/docs/streaming)的原生 `event_type`、JSON 与 `event_id`。读到 `[DONE]` 才视为完整流；提前 EOF 返回中断错误，并保留已收到的账户绑定 interaction 引用。使用 `reference()` 和 `last_event_id()` 保存引用及最新游标，然后调用 `resume_stream()` 继续读取。该调用只发起一次 `GET /interactions/{id}?stream=true&last_event_id=...`，不会自动重试。

`cancel()` 可取消仍在运行的后台 interaction，`delete()` 会删除服务端记录。两者都要求账户绑定的 `InteractionRef`。Google 没有说明事件游标保留时间；过期或无效游标会作为 provider 错误返回。测试使用本地 mock，没有调用真实 Google 账户。
