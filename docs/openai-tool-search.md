# OpenAI Responses Tool Search

OpenAI Responses Tool Search 会按需把函数或 MCP 工具定义加载到模型上下文。OpenAI 当前将此功能限定为 GPT-5.4 及更新模型。本客户端通过类型化的 `OpenAiHostedTool::ToolSearch` 同时支持 OpenAI 托管搜索和客户端执行搜索。

## OpenAI 托管搜索

将函数工具或 MCP 服务标记为延迟加载，再添加由 OpenAI 执行的搜索工具：

```rust,no_run
use lingxi_llm_client::providers::openai::types::{OpenAiToolSearchConfig, RemoteMcpConfig};
use lingxi_llm_client::protocol::{ChatRequest, LlmError};

fn configure(mut request: ChatRequest) -> Result<ChatRequest, LlmError> {
    request.hosted_tools.push(lingxi_llm_client::providers::openai::native::OpenAiHostedTool::ToolSearch(
        OpenAiToolSearchConfig::default(),
    ).into());
    for tool in &mut request.tools {
        tool.defer_loading = true;
    }

    let mcp = RemoteMcpConfig::new("orders", "https://mcp.example.test/mcp")?
        .with_defer_loading(true);
    request.hosted_tools.push(lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp).into());
    Ok(request)
}
```

Responses 请求会包含 `{"type":"tool_search"}` 及相应的延迟标记。由服务端完成发现。响应中的 `tool_search_call` 和 `tool_search_output` 保留为原生 `ProviderContent`，不会重复转换成客户端工具调用。只有响应随后返回 `function_call` 时，加载后的函数才成为客户端工具调用。

## 客户端执行搜索

当工具列表取决于应用或租户状态时，使用客户端执行模式。搜索工具需要一段说明和用于搜索参数的 JSON Schema：

```rust,no_run
use lingxi_llm_client::providers::openai::types::{OpenAiToolSearchConfig, OpenAiToolSearchExecution};
use lingxi_llm_client::protocol::{ChatRequest, };
use serde_json::json;

fn configure(mut request: ChatRequest) -> ChatRequest {
    request.hosted_tools.push(lingxi_llm_client::providers::openai::native::OpenAiHostedTool::ToolSearch(
        OpenAiToolSearchConfig {
            execution: OpenAiToolSearchExecution::Client,
            description: Some("查找完成当前任务所需的工具".into()),
            parameters: Some(json!({
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
                "additionalProperties": false
            })),
        },
    ).into());
    request
}
```

当 OpenAI 返回 `execution: "client"` 的 `tool_search_call` 时，解码响应会保留包含 `call_id` 的完整原生项，并返回 `StopReason::Other("requires_action")`。宿主执行搜索后，应发送带相同 `call_id`、`status: "completed"` 和发现到的延迟函数定义的 `tool_search_output`。将这两个原生项按顺序保留在 assistant 历史中，以便下一个 Responses 请求重放。

## 支持范围与重试

只有官方 OpenAI Responses profile 和 GPT-5.4 及更新的请求模型可以使用该功能。服务端执行模式至少需要一个延迟函数或 MCP 服务。延迟函数也必须带有对应的搜索工具。其他 codec 会拒绝 OpenAI 专属配置。

工具搜索请求固定到当前选中的连接。传输失败时，客户端不会在备用连接上重放请求，因为 OpenAI 可能已经完成了搜索。调用方会收到传输错误，并自行决定是否重试。

Responses 流解码器会将 `tool_search_call` 和 `tool_search_output` 作为原生 stream event 发出。只有 `function_call` 项会成为 `ToolCallDelta`。

官方参考：[OpenAI Tool Search 指南](https://developers.openai.com/api/docs/guides/tools-tool-search)。
