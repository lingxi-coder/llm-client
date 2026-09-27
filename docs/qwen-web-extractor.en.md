# Qwen Responses Hosted Web Extractor

[中文](qwen-web-extractor.md)

Qwen Model Studio's OpenAI-compatible Responses API provides hosted web extraction. Each request must enable both web search and web extraction: Model Studio uses search to find pages, then extracts their content for the model. Both are provider-hosted tools, not functions for the client host to execute.

```rust,ignore
use lingxi_llm_client::protocol::{ChatRequest, HostedTool, WebSearchConfig};

let mut request: ChatRequest = /* your request */;
request.hosted_tools.push(HostedTool::WebSearch(WebSearchConfig::default()));
request.hosted_tools.push(HostedTool::WebExtractor);

let response = client.chat().complete(&request, &options).await?;
// Qwen's web_extractor_call remains OpenAiResponses ProviderContent.
```

The adapter currently supports `qwen3.8-max` and `qwen3.8-flash` on Qwen Responses profiles that declare `extra.web_search = "qwen"`. Before sending, the client checks the paired search tool, tool choice, and thinking settings, then adds the documented `enable_thinking: true` flag. A request without web search, with thinking explicitly disabled, or using another model or route fails locally.

For non-streaming responses, the `web_extractor_call` item is retained as assistant `ProviderContent`, including the provider's `goal` and `output` fields. In a stream, the complete `response.output_item.done` item is emitted as `StreamEvent::ProviderContent`. The client does not parse extraction output or synthesize status fields.

When a response includes `usage.x_tools.web_extractor.count`, the client exposes it as `Usage.server_tool_usage.web_extractor_requests`. Missing or non-integer values remain `None`; the count is reported separately from tokens and is never inferred from output items.

The implementation follows Alibaba's first-party [Web Extractor guide](https://help.aliyun.com/en/model-studio/web-extractor) and [Qwen Responses API guide](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-responses). This module implements the Responses API tool form; the Chat Completions guide describes a separate streaming parameter set, which is not mixed into Responses requests. Tests use mocks only and do not call Model Studio.
