# Anthropic tool search

[简体中文](anthropic-tools.md)

Anthropic Messages offers hosted tool-catalog search with regex or BM25. Select it in `ChatRequest.hosted_tools` with `AnthropicToolSearch`, then set `ToolSpec.defer_loading: true` on function tools that should be discovered on demand:

```rust
use lingxi_llm_client::protocol::{
    AnthropicToolSearchConfig, AnthropicToolSearchStrategy, HostedTool, ToolSpec,
};
# let mut request: lingxi_llm_client::protocol::ChatRequest = serde_json::from_value(serde_json::json!({"model":"claude-opus-5-5","messages":[]})).unwrap();
request.hosted_tools.push(HostedTool::AnthropicToolSearch(
    AnthropicToolSearchConfig { strategy: AnthropicToolSearchStrategy::Bm25 },
));
request.tools.push(ToolSpec {
    name: "find_calendar_events".into(),
    description: "Find calendar events by date, person, or topic".into(),
    input_schema: serde_json::json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
    strict: false,
    defer_loading: true,
    allowed_callers: vec![],
});
```

`Regex` encodes as `tool_search_tool_regex_20251119` and searches names, descriptions, and argument fields with Python-style regular expressions. `Bm25` encodes as `tool_search_tool_bm25_20251119` and searches with natural-language queries. The request must still include every complete function-tool definition. `defer_loading` controls whether a definition is initially placed in model context; it does not omit that definition from the request. The hosted search tool itself always remains visible.

The Claude API can combine this top-level search tool and deferred catalog with a mid-conversation reference addition for a known deferred tool. Keep the full definition in `ChatRequest.tools` with `defer_loading: true`; append a system `tool_addition` that references its name when it should become available at that point:

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

The codec keeps Tool Search and the searchable definitions in the top-level `tools` array, and uses the first-party inline-tools beta for the reference change. Tool Search's guide says searchable definitions remain in `tools`. Although the mid-conversation guide allows a by-value inline definition to carry `defer_loading`, it does not say that such a message-only definition enters the hosted search catalog; this guide makes no such discoverability claim.

Anthropic allows at most 10,000 deferred tools per request. Regex patterns are limited to 200 characters and BM25 queries to 500 characters. The model generates those queries at runtime, and Anthropic validates them.

The decoder preserves `server_tool_use` and `tool_search_tool_result` blocks as `ContentBlock::ProviderContent` so the assistant message can be replayed unchanged in the next request. Streaming emits the complete native search-call block after its input fragments arrive. Discovered ordinary function calls remain `ToolUse` / `ToolCallDelta` events for the host to execute; the client does not search the catalog or execute functions.

This option accepts the first-party Anthropic Messages API, the Foundry Anthropic Messages route, Google Cloud Vertex Claude, and Amazon Bedrock InvokeModel, including streaming. Foundry Tool Search requires an explicit `FoundryDeployment` on the matching `ModelProfile` row; `request.model` remains the custom deployment name. Only exact hosting/model pairs in the intersection of Foundry availability and Anthropic’s Tool Search compatibility table are accepted. AnthropicMessages also requires `provider_id` to be `anthropic`; Bedrock has no Converse route. OpenAI and Gemini fail before HTTP with `UnsupportedCapability`. `ToolChoice` is encoded as requested.

`defer_loading` is Anthropic-specific; other codecs reject it rather than silently treating it as an ordinary tool parameter. The Claude API, Vertex, and Foundry each validate exact model IDs against their documented compatibility tables; the per-host Foundry intersection is listed in the [Foundry guide](anthropic-foundry.en.md). Sonnet 5 is not listed for Tool Search. Account and regional availability remain provider-authoritative. Bedrock model IDs and inference-profile ARNs pass through unchanged; the client does not infer the model from an opaque ARN.

Anthropic’s current Tool Search table lists Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 5, Opus 4.8/4.7/4.6/4.5, Sonnet 4.6/4.5, and Haiku 4.5; Opus 4.1 and earlier are unsupported. The codec validates exact model IDs separately for the Claude API, Vertex, and Foundry instead of accepting display names or arbitrary suffixes. For Foundry, use only the host/model intersection in the [Foundry guide](anthropic-foundry.en.md), including undated 4.5 IDs. `claude-sonnet-5` is absent from the Tool Search table.

## Official documentation

- [Anthropic Foundry guide](anthropic-foundry.en.md)

- [Anthropic Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Anthropic mid-conversation system messages and tool changes](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- [Anthropic Tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)

Local wire fixtures verify request encoding, native-block replay, and stream assembly. They do not establish availability on a live account or model.

See [Preserving Anthropic-native content](anthropic-native-content.en.md) for preservation behavior for other server-side blocks, future block types, and stream events. It also explains the boundary between retaining native blocks and executing client tools.

Vertex uses the published `@YYYYMMDD` version IDs for older models, rather than the Claude API dated form. See the [Vertex guide](anthropic-vertex.en.md).
