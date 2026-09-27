# OpenAI Responses tool search

OpenAI Responses tool search loads function or MCP definitions only when they are needed. OpenAI documents it for GPT-5.4 and later. This client supports both OpenAI-hosted discovery and client-executed discovery through the typed `HostedTool::OpenAiToolSearch` configuration.

## OpenAI-hosted discovery

Mark the function tools or MCP servers to defer, then add the server-executed search tool:

```rust,no_run
use lingxi_llm_client::protocol::{
    ChatRequest, HostedTool, OpenAiToolSearchConfig, LlmError, RemoteMcpConfig,
};

fn configure(mut request: ChatRequest) -> Result<ChatRequest, LlmError> {
request.hosted_tools.push(HostedTool::OpenAiToolSearch(
    OpenAiToolSearchConfig::default(),
));
for tool in &mut request.tools {
    tool.defer_loading = true;
}

let mcp = RemoteMcpConfig::new("orders", "https://mcp.example.test/mcp")?
    .with_defer_loading(true);
request.hosted_tools.push(HostedTool::RemoteMcp(mcp));
Ok(request)
}
```

The Responses request contains `{"type":"tool_search"}` and the deferred flags. The provider performs discovery. Its `tool_search_call` and `tool_search_output` items remain native `ProviderContent` in the response history; they do not become duplicate client tool calls. A loaded function is a client call only when the response later contains a `function_call` item.

## Client-executed discovery

Use client execution when the inventory depends on application or tenant state. The search tool needs a description and a JSON Schema for its arguments:

```rust,no_run
use lingxi_llm_client::protocol::{
    ChatRequest, HostedTool, OpenAiToolSearchConfig, OpenAiToolSearchExecution,
};
use serde_json::json;

fn configure(mut request: ChatRequest) -> ChatRequest {
request.hosted_tools.push(HostedTool::OpenAiToolSearch(
    OpenAiToolSearchConfig {
        execution: OpenAiToolSearchExecution::Client,
        description: Some("Find the tools needed for this task".into()),
        parameters: Some(json!({
            "type": "object",
            "properties": {"goal": {"type": "string"}},
            "required": ["goal"],
            "additionalProperties": false
        })),
    },
));
request
}
```

When OpenAI returns a `tool_search_call` with `execution: "client"`, the decoded response keeps the complete native item, including `call_id`, and reports `StopReason::Other("requires_action")`. The host performs discovery and sends a `tool_search_output` with the same `call_id`, `status: "completed"`, and the discovered deferred function definitions. Keep both native items in the assistant history so the next Responses request can replay them in order.

## Scope and retry behavior

Tool search is accepted only on the official OpenAI Responses profile and with a GPT-5.4-or-later request model. Server execution requires at least one deferred function or MCP server. Deferred function flags still require a matching search tool. Other codecs reject the OpenAI-specific configuration.

The client pins a tool-search request to its selected connection. It does not replay the request on a fallback after a transport failure, since the provider may already have performed discovery. The caller receives the transport error and decides whether to retry.

The shared Responses decoder emits `tool_search_call` and `tool_search_output` as native stream events. Only `function_call` items become `ToolCallDelta` events.

Official reference: [OpenAI Tool Search guide](https://developers.openai.com/api/docs/guides/tools-tool-search).
