# LLM Web Search

[简体中文](web-search.md)

`LlmClient::web_search()` and `web_search_stream()` enable provider-hosted search for a single request without implementing a search function or registering a client-side `ToolSpec`. You can also set `CompletionRequest.web_search` and then call `client.chat().complete()` / `client.chat().stream()`; it defaults to `None`, preserving the original request behavior. The server decides whether to search; enabling search does not guarantee a search on every request.

## Quick start

The built-in `openai`, `anthropic`, `gemini`, `openrouter`, `zai`, `glm`, `kimi-search`, `kimi-search-intl`, Qwen Search, and MiniMax profiles declare search adapters. With a model that supports search, set the following on an existing request:

```rust
use lingxi_llm_client::protocol::WebSearchConfig;
# let mut request: lingxi_llm_client::protocol::CompletionRequest = serde_json::from_value(serde_json::json!({"model": "gpt-4.1", "messages": []})).unwrap();
request.web_search = Some(WebSearchConfig::default());
```

When you know which connection owns the credentials, use `client.chat().complete_in("openai", &request, &options)`. You can also pass the configuration separately to `client.web_search_in("openai", &request, WebSearchConfig::default(), &options)`; the streaming equivalent is `web_search_stream_in()`. `request.model` can be the connection's native model ID. The search methods do not modify `request` and override any `web_search` configuration already in it. The connection-agnostic `complete()`, `web_search()`, and `web_search_stream()` remain available, but return an ambiguity error when a native model ID and `profile/model` could be interpreted differently. The provider validates whether search is available for a particular model, account, and deployment. A profile declaration only indicates which search interface the connection uses; it does not mean every model in its catalog supports search.

To restrict sources or the number of searches:

```rust
use lingxi_llm_client::protocol::WebSearchConfig;
let search = WebSearchConfig {
    allowed_domains: vec!["rust-lang.org".into()],
    ..WebSearchConfig::default()
};
```

## Support matrix and parameters

| `extra.web_search` | Protocol | Encoding | Supported additional options |
| --- | --- | --- | --- |
| `openai_responses` | `open_ai_responses` | `tools: [{type: "web_search"}]` | `allowed_domains`, up to 100 |
| `qwen` | `open_ai_responses` | `tools: [{type: "web_search"}]` | `allowed_domains`, up to 100; no `blocked_domains` |
| `openai_chat` | `open_ai_chat` | `web_search_options: {}` | None; requires a dedicated search model |
| `anthropic` | `anthropic_messages`, `foundry_claude`, `vertex_claude` | `web_search_20250305` | `allowed_domains` or `blocked_domains`; positive `max_uses` |
| `minimax` | `anthropic_messages` | `web_search_20250305` | No domain filters or `max_uses`; `ToolChoice` supports `Auto` / `None` only |
| `gemini` | `gemini_generate_content`, `vertex_gemini` | `tools: [{googleSearch: {}}]` | None |
| `openrouter` | `open_ai_chat` | `tools: [{type: "openrouter:web_search"}]` | `allowed_domains` or `blocked_domains` |
| `glm` | `open_ai_chat` | `tools: [{type: "web_search", web_search: {...}}]` | One `allowed_domains` domain; engine specified by the profile |
| `kimi` | `open_ai_responses` | `web_search`, also requesting `action.sources` | `allowed_domains`, up to 100 |
| `deepseek` | `anthropic_messages` | Basic Anthropic-compatible search format | None; unconfirmed filtering and usage limits are rejected |
| `xai` | `open_ai_responses` | `tools: [{type: "web_search"}]` | `allowed_domains` or `blocked_domains`, up to 5 |

Example models, connection URLs, and credential sources for built-in profiles are listed below. For a custom profile, `extra.web_search` must match its protocol. Serialized `WebSearchConfig` is also part of the JSON interface, for example `{"model":"glm/glm-4.7","messages":[...],"web_search":{"allowed_domains":["example.com"]}}`. Domain parameters must contain only domain names, without a scheme, path, or spaces. An empty configuration `{}` is equivalent to `WebSearchConfig::default()`.

The common options accept only domain names without a protocol prefix or path; allowlists and blocklists cannot be set together. Unsupported options return `UnsupportedCapability`, and malformed values return `InvalidRequest`. Regular tools and search tools are merged, so existing tools are preserved. The Gemini, Chat search, GLM, Kimi, Qwen, and DeepSeek search adapters require `ToolChoice::Auto`; MiniMax accepts only `Auto` / `None`. Other interfaces retain choices such as `None` and `Any`. `None` forbids tool execution for the current request.

Custom connections must explicitly declare `profile.extra["web_search"]`. An undeclared or unknown adapter, a protocol mismatch, or a search request through Bedrock produces an error before any network request is sent. Every failover connection is validated again; the search requirement is not automatically removed.

The existing xAI `grok` preset uses Chat Completions and keeps that default route. For Grok search, clone it into a separate connection:

```rust
use lingxi_llm_client::protocol::{DirectoryRoute, ProtocolFamily};
# let profiles = lingxi_llm_client::builtin_providers().unwrap();
let mut profile = profiles.iter()
    .find(|p| p.profile_name == "grok").unwrap().clone();
profile.profile_name = "grok-search".into();
profile.protocol = ProtocolFamily::OpenAiResponses;
profile.model_list = DirectoryRoute::Shape(ProtocolFamily::OpenAiChat);
profile.extra["web_search"] = serde_json::json!("xai");
```

Pass this profile to the builder and request `grok-search/<model ID>`. The provider identity and credentials do not need to change. Do not treat other vendors that only support the chat format as search-capable endpoints.

## DeepSeek, GLM, and Kimi

These connections use different APIs and do not automatically modify the existing chat or Coding Plan connections. Search still requires explicitly setting `web_search`:

| Example request model | Connection URL | Credential environment variable | Notes |
| --- | --- | --- | --- |
| `deepseek-search/deepseek-flash` | `https://api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` | Also supports `deepseek-v4-pro` in the catalog |
| `glm/glm-4.7` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | Mainland China open platform, usage-based billing |
| `zai/glm-4.7` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` | International platform, separate account |
| `kimi-search/kimi-k3` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | Responses API; current official documentation lists only K3 |

```rust
use lingxi_llm_client::protocol::{CompletionRequest, ConversationMessage, WebSearchConfig};
let mut request: CompletionRequest = serde_json::from_value(serde_json::json!({
    "model": "kimi-search/kimi-k3",
    "messages": []
})).unwrap();
request.messages.push(ConversationMessage::user_text("搜索今天的科技新闻并注明来源"));
request.web_search = Some(WebSearchConfig::default());
```

DeepSeek's official [Claude Code integration documentation](https://api-docs.deepseek.com/zh-cn/quick_start/agent_integrations/claude_code/) states that its Anthropic endpoint supports native search, while its [Responses compatibility table](https://api-docs.deepseek.com/guides/responses_api/) explicitly lists `web_search` as ignored. Therefore, search is integrated only through a separate Anthropic connection here. Sending `web_search_20250305` is an inference based on Claude tool compatibility: DeepSeek does not separately document a tool version or advanced options, and this has not yet been verified with real credentials. The existing `deepseek` Chat connection continues to reject common search options. The DeepSeek search connection permits replay of unsigned thinking blocks; other Anthropic connections still require the original signatures.

## China and international profiles

Connections for a provider's regions use distinct endpoints and keys. Choose the profile for the region where the key was created; do not configure a key from another region. Ordinary Chat and hosted search remain separate connections:

| Service and region | Chat profile / URL / credential variable | Search profile / URL / credential variable |
| --- | --- | --- |
| Qwen Beijing | `qwen` · `https://dashscope.aliyuncs.com/compatible-mode/v1` · `DASHSCOPE_API_KEY` | `qwen-search` · same host · `DASHSCOPE_API_KEY` |
| Qwen Singapore | `qwen-intl` · `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` · `DASHSCOPE_INTL_API_KEY` | `qwen-search-intl` · same host · `DASHSCOPE_INTL_API_KEY` |
| Qwen US Virginia | `qwen-us` · `https://dashscope-us.aliyuncs.com/compatible-mode/v1` · `DASHSCOPE_US_API_KEY` | `qwen-search-us` · same host · `DASHSCOPE_US_API_KEY` |
| Qwen Hong Kong | `qwen-hk` · `https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1` · `DASHSCOPE_HK_API_KEY` | `qwen-search-hk` · same host · `DASHSCOPE_HK_API_KEY` |
| MiniMax China | `minimax` · `https://api.minimaxi.com/anthropic` · `MINIMAX_CN_API_KEY` | `minimax` · same host · `MINIMAX_CN_API_KEY` |
| MiniMax international | `minimax-intl` · `https://api.minimax.io/anthropic` · `MINIMAX_API_KEY` | `minimax-intl` · same host · `MINIMAX_API_KEY` |
| Kimi international | `kimi-intl` · `https://api.moonshot.ai/v1` · `MOONSHOT_INTL_API_KEY` | `kimi-search-intl` · same host · `MOONSHOT_INTL_API_KEY` |

`qwen-search` and `qwen-search-intl` also declare Qwen Responses knowledge-base retrieval. Each request needs a `FileSearchConfig { knowledge_base_id, workspace_id }`. Qwen's file-search request uses the region's workspace-specific Model Studio domain. Search profiles in the other Qwen regions declare Web Search only.

GLM uses `enable: true` and `search_result: true`, with `search_pro` as the default engine for the mainland China platform and `search-prime` for Z.AI. A custom connection can specify an engine available to its account with `extra.web_search_engine`. Raw `web_search` entries from the response are retained in metadata; `link/title` become the source list, while `refer`, summaries, and publication dates are preserved. Streaming search frames without choices are also preserved. `glm-coding` does not declare this capability because Coding Plan's MCP search is a different API from the chat API.

Kimi uses server-side search through its [current Responses API](https://platform.kimi.com/docs/api/responses); the client does not need to execute or return a search tool call. `include: ["web_search_call.action.sources"]` requests native sources. Sources are mapped to `citations`, and the complete search action remains in metadata. Search mode does not accept `temperature` or the common thinking token budget. The old `$web_search` built-in interface is being deprecated and is not connected to the `kimi` Chat or `kimi-code` subscription connections.

The new connections use separate connection groups to avoid automatically falling back from a search request to ordinary chat, subscription, or another region's account. DeepSeek and Kimi token prices use the repository's existing catalog snapshot. The usage-billed `glm` has no verified price in US dollars, so `estimate_cost()` returns `CostUnavailable` rather than incorrectly applying the zero price of Coding Plan to the open platform. Search charges are still excluded from token estimates.

## Search results and citations

For a regular response, `response.web_search: Option<WebSearchResult>` contains `citations` (URLs and optional titles) and `metadata` (native search metadata). It is `None` when there is no search metadata. Text is still obtained through `response.message.text()`.

```json
{
  "message": {"role": "assistant", "content": [{"type": "text", "text": "答案..."}]},
  "web_search": {
    "citations": [{"url": "https://example.com/article", "title": "来源标题"}],
    "metadata": {"web_search": [{"link": "https://example.com/article", "title": "来源标题", "refer": "ref_1"}]}
  },
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 0, "cache_write_tokens": 0},
  "model": "glm-4.7"
}
```

This illustrates the field structure. The contents of `metadata` vary by provider and response format; this example shows GLM's `web_search` field and is not a contract for parsing a fixed format. The only guaranteed standard source fields are `url` and optional `title`.

Streaming responses add `StreamEvent::WebSearch { result }` for search records received in that frame; it is not a `ToolCallDelta` to execute. Preserve these events rather than consuming only `TextDelta`. Metadata across frames is not a common cumulative object; Gemini grounding data is preserved as the snapshot returned by the provider in each frame. You can deduplicate sources by URL for display, but retain native citation positions and associations.

Native metadata preserves OpenAI annotations, Claude citations and search results (including search failure details), and Gemini grounding supports, queries, and Search Suggestions. Each provider defines its own position units and block indices for citations; they are not uniformly converted to Rust byte offsets. When presenting answers to users, use these fields to show clickable citations. Gemini also requires displaying the returned Search Suggestions. The client does not generate an HTML interface.

In regular Claude responses, search blocks that must be replayed unchanged and text with search citations are stored as `ContentBlock::ProviderContent`. Add the entire `response.message` to the next `request.messages` to preserve `encrypted_content`, `encrypted_index`, and the relationship to search calls. `message.text()` also includes these native text blocks, while `message.tool_uses()` does not treat server-side search as a local tool.

`StopReason::Other("pause_turn")` means Claude has paused the server-side loop. The caller can append that assistant message unchanged and send another request to continue. The client does not automatically make another billable request. Streaming callers must reconstruct text and native search content by block index before replay; concatenating text alone loses encrypted context. Replaying native content through another protocol is rejected.

Search failures may also arrive with HTTP 200, such as Claude's `max_uses_exceeded`. In that case, the search error remains in metadata and is not presented as a successful citation. Callers should inspect the search records to decide whether to retry or inform the user.

## Costs and compatibility

Search may be billed separately. The existing `estimate_cost()` estimates token costs only and excludes search call charges; do not treat it as the total bill when search is enabled. Retain search usage metadata and any reported charges returned by the provider, and use the provider's bill as the source of truth.

Old JSON requests and responses still deserialize; the new optional fields default to absent. Rust struct literals must add `web_search: None`. Downstream code that exhaustively matches `StreamEvent` / `ContentBlock` must handle the new variants.

## Official documentation

This implementation follows the official interface documentation below and uses the basic Claude search version to avoid requiring dynamic filtering and code execution:

- [OpenAI Web search](https://developers.openai.com/api/docs/guides/tools-web-search)
- [Anthropic Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
- [Gemini Grounding with Google Search (generateContent)](https://ai.google.dev/gemini-api/docs/generate-content/google-search)
- [xAI Web Search](https://docs.x.ai/developers/tools/web-search)
- [DeepSeek Anthropic API](https://api-docs.deepseek.com/guides/anthropic_api/)
- [GLM Web Search](https://docs.bigmodel.cn/cn/guide/tools/web-search)
- [Z.AI Web Search in Chat](https://docs.z.ai/guides/tools/web-search)
- [Kimi Responses API](https://platform.kimi.com/docs/api/responses)
- [OpenRouter Web Search Server Tool](https://openrouter.ai/docs/guides/features/server-tools/web-search)

Verification used local fixtures in official formats and simulated transport, without real API keys. It does not establish account permissions, model availability, or actual charges.
