# Qwen Responses Hosted Code Interpreter

[中文](qwen-hosted.md)

Qwen's OpenAI-compatible Responses API can invoke Model Studio's hosted Python interpreter with `HostedTool::CodeInterpreter`. The client sends the documented `{"type":"code_interpreter"}` tool and `enable_thinking: true`; Alibaba runs the code, and this crate never executes it locally.

```rust,ignore
use lingxi_llm_client::protocol::{
    ChatRequest, CodeInterpreterConfig, HostedTool,
};

let mut request: ChatRequest = /* your request */;
request.hosted_tools.push(HostedTool::CodeInterpreter(
    CodeInterpreterConfig::default(),
));

let response = client.chat().complete(&request, &options).await?;
// Qwen's code_interpreter_call is retained as OpenAiResponses ProviderContent;
// its code, execution output, container_id, and native status remain available.
```

The adapter is currently enabled only for `qwen3.8-max` and `qwen3.8-flash` on Qwen Responses profiles that declare the Qwen web-search adapter. Before sending, it rejects OpenAI container memory settings, combinations with caller-defined function tools, non-automatic tool choice, and requests that explicitly disable thinking. The Responses request supplies the required `enable_thinking: true` flag. Qwen's Web Extractor is exposed separately as `HostedTool::WebExtractor` and requires `HostedTool::WebSearch`; see [Qwen Web Extractor](qwen-web-extractor.en.md).

For non-streaming responses, the `code_interpreter_call` remains in the assistant message as `ProviderContent`. Streaming responses retain the complete call item and emit each `response.code_interpreter_call.in_progress`, `interpreting`, and `completed` event as native `ProviderContent`. Callers can inspect the status, generated `code`, `outputs` logs, and `container_id`. The client does not interpret logs, execute code, or turn this provider-hosted call into a host-executed `ToolUse`.

When a response contains `usage.x_tools.code_interpreter.count`, `Usage.server_tool_usage.code_interpreter_requests` exposes the value. Missing or non-integer counts remain `None`; the client does not infer a count from output items. This measure stays separate from token totals and is not used to estimate cost.

The implementation follows Alibaba's first-party [Qwen Code Interpreter guide](https://help.aliyun.com/en/model-studio/qwen-code-interpreter) and [Qwen Responses API guide](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-responses). Tests use mocks only and do not call Model Studio.

Requests containing Code Interpreter are dispatched once to the first connection. Failure does not trigger connection failover or automatic replay to repair an expired attachment. Code may already have executed in the container; the caller decides whether to submit again.
