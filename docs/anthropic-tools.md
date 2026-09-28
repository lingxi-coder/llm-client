# Anthropic 工具搜索

[English](anthropic-tools.en.md)

Anthropic Messages 支持 regex 和 BM25 两种托管工具目录搜索。将搜索方式设为 `ChatRequest.hosted_tools` 的 `AnthropicToolSearch`，再对需要按需发现的函数工具设置 `ToolSpec.defer_loading: true`：

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicToolSearchConfig, AnthropicToolSearchStrategy};
use lingxi_llm_client::protocol::{ToolSpec};
# let mut request: lingxi_llm_client::protocol::ChatRequest = serde_json::from_value(serde_json::json!({"model":"claude-opus-5-5","messages":[]})).unwrap();
request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
    AnthropicToolSearchConfig { strategy: AnthropicToolSearchStrategy::Bm25 },
).into());
request.tools.push(ToolSpec {
    tool_type: None,
    extra: serde_json::Value::Null,
    name: "find_calendar_events".into(),
    description: "Find calendar events by date, person, or topic".into(),
    input_schema: serde_json::json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
    strict: false,
    defer_loading: true,
    native_options: Vec::new(),
});
```

`Regex` 编码为 `tool_search_tool_regex_20251119`，使用 Python 风格正则匹配名称、说明及参数；`Bm25` 编码为 `tool_search_tool_bm25_20251119`，使用自然语言查询。启用搜索后，请求仍须携带所有函数工具的完整定义。`defer_loading` 只控制工具定义是否立即进入模型上下文，不会省略请求中的定义。托管搜索工具本身始终保持可见；不要把所有工具都延迟。

Claude API 可将顶层 hosted Tool Search 和延迟目录，与会话中途按引用添加已知工具组合使用。完整定义继续留在 `ChatRequest.tools` 并设置 `defer_loading: true`；需要从某个位置开始提供时，再追加引用该名称的 system `tool_addition`：

```json
{
  "role": "system",
  "content": [
    {
      "type": "tool_addition",
      "tool": {"type": "tool_reference", "name": "find_calendar_events"}
    }
  ]
}
```

codec 会把 Tool Search 和可搜索的完整定义保留在顶层 `tools` 数组，并为第一方引用变更添加 inline-tools beta。Tool Search 文档要求可搜索定义仍位于 `tools`。虽然中途工具变更指南允许按值定义携带 `defer_loading`，但没有说明仅出现在消息中的定义会加入 hosted Tool Search 目录；本文不宣称这种发现行为。

官方限制每个请求最多 10,000 个延迟加载工具；Regex 搜索模式最多 200 个字符，BM25 搜索查询最多 500 个字符。这些查询由模型在运行时生成并由 Anthropic 校验。

搜索工具返回的 `server_tool_use` 和 `tool_search_tool_result` 块会保留在 `ContentBlock::ProviderContent` 中，以便把 assistant 消息原样放回下一轮请求。流式解码也会在接收完搜索输入片段后发出完整的原生调用块。发现的普通函数调用仍通过 `ToolUse` / `ToolCallDelta` 交给宿主执行；客户端不会自行搜索目录或执行这些函数。

此选项接受 Anthropic 官方 Messages API、Foundry 的 Anthropic Messages 路由、Vertex Claude，以及 Amazon Bedrock InvokeModel（包括流式调用）。Foundry Tool Search 需在对应 `ModelProfile` 行显式填写 `FoundryDeployment`；请求中的 `model` 仍是自定义部署名。只有 Foundry 托管方式与官方 Tool Search 模型表的精确交集可用。AnthropicMessages 还要求 profile 的 `provider_id` 为 anthropic；Bedrock 没有 Converse 路径。OpenAI 和 Gemini 会在发 HTTP 前返回 `UnsupportedCapability`。`ToolChoice` 按请求原样编码。

`defer_loading` 是 Anthropic 专用字段；其他 `codec` 会拒绝将它静默当作普通工具参数。Claude API、Vertex 和 Foundry 均按各自官方兼容表核验精确模型 ID；Foundry 的分 hosting 列表见[Foundry 指南](anthropic-foundry.md)。Sonnet 5 不在 Tool Search 表中。账户和地区可用性仍由服务端决定。Bedrock 的模型 ID 和 inference-profile ARN 原样保留，不从不透明 ARN 推断模型。

Anthropic 当前列出的模型包括 Claude Fable 5.1、Mythos 5.1、Fable 5、Mythos 5、Opus 5.5、Opus 5、Opus 4.8、4.7、4.6、4.5、Sonnet 4.6、4.5 和 Haiku 4.5；Opus 4.1 及更早版本不支持。此列表会随官方兼容矩阵变化；codec 依据官方表核验 Claude API / Vertex 的 wire ID，不能用显示名称或任意后缀代替。Sonnet 5 不在当前 Tool Search 表中；它的 Browser/Computer 工具集可以直接使用，但延迟搜索需选择已确认支持 Tool Search 的模型。

## 官方文档

- [Anthropic Foundry 指南](anthropic-foundry.md)

- [Anthropic Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Anthropic Mid-conversation system messages and tool changes](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- [Anthropic Tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)

本地 wire fixtures 验证请求结构、原生块重放及流式拼接，不代表已验证真实账户或模型可用性。

关于其他原生服务端块、未来块类型及流事件的保留边界，见[Anthropic 原生内容保留](anthropic-native-content.md)。该文档也说明原生块保留与客户端工具执行的区别。

Vertex 使用官方列出的 `@YYYYMMDD` 旧款模型版本 ID，不把 Claude API 的日期形式直接套用到云端。参见 [Vertex 指南](anthropic-vertex.md)。
