# Qwen Responses 托管代码解释器

[English](qwen-hosted.en.md)

Qwen 的 OpenAI 兼容 Responses API 可通过 `OpenAiHostedTool::CodeInterpreter` 调用百炼托管的 Python 代码解释器。客户端按文档发送 `{"type":"code_interpreter"}` 和 `enable_thinking: true`；代码运行由百炼托管，客户端不会本地执行代码。

```rust,ignore
use lingxi_llm_client::providers::openai::types::{CodeInterpreterConfig};
use lingxi_llm_client::protocol::{ChatRequest, };

let mut request: ChatRequest = /* 你的请求 */;
request.hosted_tools.push(lingxi_llm_client::providers::openai::native::OpenAiHostedTool::CodeInterpreter(
    CodeInterpreterConfig::default(),
).into());

let response = client.chat().complete(&request, &options).await?;
// Qwen 的 code_interpreter_call 作为 OpenAiResponses ProviderContent 保留；
// 完整代码、执行输出、container_id 和原生状态不会被压平成文本。
```

当前仅对声明 Qwen web-search adapter 的 Qwen Responses profile 中的 `qwen3.8-max` 和 `qwen3.8-flash` 启用。客户端会在发送前拒绝 OpenAI 容器内存配置、与自定义函数工具混用、非自动工具选择，以及显式关闭 thinking 的请求。Responses profile 会自动携带此工具要求的 `enable_thinking: true`。Qwen 网页抓取作为独立的 `QwenHostedTool::WebExtractor` 暴露，并要求同时配置 `HostedTool::WebSearch`，详见 [Qwen 网页抓取](qwen-web-extractor.md)。

Qwen 非流式返回中的 `code_interpreter_call` 保留在 assistant 消息的 `ProviderContent` 中；流式响应保留完整调用项，并把 `response.code_interpreter_call.in_progress`、`interpreting`、`completed` 事件逐个作为原生 `ProviderContent` 输出。调用方可读取状态、生成的 `code`、`outputs` 日志和 `container_id`。客户端不解释日志、不执行代码，也不把托管调用伪装成需要主机执行的 `ToolUse`。

当响应包含 `usage.x_tools.code_interpreter.count` 时，`Usage.server_tool_usage.code_interpreter_requests` 暴露该计数。字段缺失或不是非负整数时保持 `None`，不会用调用项数量推算。该计数与 token 数分开报告，不用于估算费用。

实现依据阿里云第一方 [Qwen Code Interpreter 文档](https://help.aliyun.com/en/model-studio/qwen-code-interpreter)和 [Qwen Responses API 文档](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-responses)。测试仅使用 mock，不访问百炼账户。

包含 Code Interpreter 的请求只派发到首选连接一次；失败不会自动切换连接，也不会因附件失效而自动重发。容器中可能已执行代码，是否重新提交由调用方决定。
