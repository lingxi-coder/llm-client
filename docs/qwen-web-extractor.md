# Qwen Responses 托管网页抓取

[English](qwen-web-extractor.en.md)

Qwen Model Studio 的 OpenAI 兼容 Responses API 提供托管网页抓取。请求必须同时启用网页搜索和网页抓取：百炼先用搜索定位网页，再抓取网页内容供模型使用。两者都是服务端工具，不会转成需要宿主执行的函数调用。

```rust,ignore
use lingxi_llm_client::protocol::{ChatRequest, HostedTool, WebSearchConfig};

let mut request: ChatRequest = /* 你的请求 */;
request.hosted_tools.push(HostedTool::WebSearch(WebSearchConfig::default()));
request.hosted_tools.push(HostedTool::WebExtractor);

let response = client.chat().complete(&request, &options).await?;
// Qwen 的 web_extractor_call 作为 OpenAiResponses ProviderContent 保留。
```

当前仅对 `qwen3.8-max` 和 `qwen3.8-flash` 的 Qwen Responses profile 启用；profile 需要声明 `extra.web_search = "qwen"`。客户端会在发送前检查网页搜索配对、工具选择及 thinking 配置，并自动写入文档要求的 `enable_thinking: true`。显式关闭 thinking、只配置网页抓取、使用其他模型或路由都会在本地报错。

非流式响应中的 `web_extractor_call` 原样保留在 assistant 消息 `ProviderContent` 中，可读取服务端返回的 `goal` 和 `output`。流式响应会把完整 `response.output_item.done` 项作为 `StreamEvent::ProviderContent` 发出。客户端不解析抓取结果，也不合成状态字段。

响应若包含 `usage.x_tools.web_extractor.count`，客户端通过 `Usage.server_tool_usage.web_extractor_requests` 暴露该调用数。字段缺失或不是非负整数时保持 `None`；计数与 token 分开报告，也不会从输出项数量推算。

实现依据阿里云第一方 [网页抓取文档](https://help.aliyun.com/en/model-studio/web-extractor)和 [Qwen Responses API 文档](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-responses)。这里实现的是 Responses API 的工具形式；Chat Completions API 文档描述的是另一组流式参数，本模块不把它们混入 Responses 请求。测试使用 mock，不访问百炼账户。
