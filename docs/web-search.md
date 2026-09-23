# LLM Web Search

`LlmClient::web_search()` 和 `web_search_stream()` 为单次请求启用 provider 托管的搜索，不需要实现搜索函数或注册客户端 `ToolSpec`。也可设置 `CompletionRequest.web_search` 后调用 `complete()` / `stream()`；默认 `None`，保持原请求行为。服务端决定是否搜索；启用不保证每次都会搜索。

## 快速使用

内置 `openai`、`anthropic`、`gemini`、`openrouter`、`zai`、`glm`、`kimi-search` 和 `deepseek-search` profile 已声明搜索适配器。使用支持搜索的模型，在已有请求上设置：

```rust
use lingxi_agent_api::protocol::WebSearchConfig;
# let mut request: lingxi_agent_api::protocol::CompletionRequest = serde_json::from_value(serde_json::json!({"model": "openai/gpt-4.1", "messages": []})).unwrap();
request.web_search = Some(WebSearchConfig::default());
```

可通过 `client.complete(&request, &options)` 发送，也可将配置作为独立参数传入 `client.web_search(&request, WebSearchConfig::default(), &options)`；流式对应 `web_search_stream()`。后两者不会修改 `request`，并覆盖其中已有的 `web_search` 配置。搜索功能是否对具体模型、账户和部署开放由 provider 校验；profile 声明仅表示该连接使用哪种搜索接口，不代表其目录中的全部模型都支持搜索。

限定来源或搜索次数：

```rust
use lingxi_agent_api::protocol::WebSearchConfig;
let search = WebSearchConfig {
    allowed_domains: vec!["rust-lang.org".into()],
    ..WebSearchConfig::default()
};
```

## 支持矩阵与参数

| `extra.web_search` | 协议 | 编码 | 支持的附加选项 |
| --- | --- | --- | --- |
| `openai_responses` | `open_ai_responses` | `tools: [{type: "web_search"}]` | `allowed_domains`，最多 100 个 |
| `openai_chat` | `open_ai_chat` | `web_search_options: {}` | 无；需要专用搜索模型 |
| `anthropic` | `anthropic_messages`、`foundry_claude`、`vertex_claude` | `web_search_20250305` | `allowed_domains` 或 `blocked_domains`；正数 `max_uses` |
| `gemini` | `gemini_generate_content`、`vertex_gemini` | `tools: [{googleSearch: {}}]` | 无 |
| `openrouter` | `open_ai_chat` | `tools: [{type: "openrouter:web_search"}]` | `allowed_domains` 或 `blocked_domains` |
| `glm` | `open_ai_chat` | `tools: [{type: "web_search", web_search: {...}}]` | 一个 `allowed_domains` 域名；引擎由 profile 指定 |
| `kimi` | `open_ai_responses` | `web_search`，同时请求 `action.sources` | `allowed_domains`，最多 100 个 |
| `deepseek` | `anthropic_messages` | 基础 Anthropic 搜索兼容格式 | 无；未确认的过滤及次数控制均拒绝 |
| `xai` | `open_ai_responses` | `tools: [{type: "web_search"}]` | `allowed_domains` 或 `blocked_domains`，最多 5 个 |

内置 profile 的示例模型、连接地址和凭证来源见下文。对自定义 profile，`extra.web_search` 必须与其协议匹配。`WebSearchConfig` 序列化后也是 JSON 接口的一部分，例如 `{"model":"glm/glm-4.7","messages":[...],"web_search":{"allowed_domains":["example.com"]}}`。域名参数只能写域名，不含 scheme、路径或空格。空配置 `{}` 等价于 `WebSearchConfig::default()`。

统一选项仅接受不含协议前缀和路径的域名；允许列表和禁止列表不能同时设置。不支持的选项返回 `UnsupportedCapability`，格式错误返回 `InvalidRequest`。普通工具与搜索工具会合并，已有工具不会被覆盖。Gemini、Chat 搜索、GLM、Kimi 和 DeepSeek 搜索适配器要求 `ToolChoice::Auto`；其他接口会保留 `None`、`Any` 等选择。`None` 表示禁止本次工具执行。

自定义连接必须显式声明 `profile.extra["web_search"]`。未声明、适配器未知、协议不匹配或 Bedrock 请求搜索时，会在发送网络请求前报错。故障转移的每条连接也会重新校验，不会自动删除搜索要求。

xAI 原有 `grok` 预设使用 Chat Completions，保留此默认路由。要使用 Grok 搜索，可从它克隆一个单独的连接：

```rust
use lingxi_agent_api::protocol::{DirectoryRoute, ProtocolFamily};
# let profiles = lingxi_llm_client::builtin_providers().unwrap();
let mut profile = profiles.iter()
    .find(|p| p.profile_name == "grok").unwrap().clone();
profile.profile_name = "grok-search".into();
profile.protocol = ProtocolFamily::OpenAiResponses;
profile.model_list = DirectoryRoute::Shape(ProtocolFamily::OpenAiChat);
profile.extra["web_search"] = serde_json::json!("xai");
```

将此 profile 交给构建器，使用 `grok-search/<模型 ID>` 请求。无需更改 provider 身份或凭证。不要将仅兼容聊天格式的其他供应商当作支持搜索的端点。

## DeepSeek、GLM 与 Kimi

这些连接使用不同的 API，不自动修改原有聊天或 Coding Plan 连接。搜索仍须显式设置 `web_search`：

| 请求模型示例 | 连接地址 | 凭证环境变量 | 说明 |
| --- | --- | --- | --- |
| `deepseek-search/deepseek-flash` | `https://api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` | 另支持目录中的 `deepseek-v4-pro` |
| `glm/glm-4.7` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | 国内开放平台，按量计费 |
| `zai/glm-4.7` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` | 国际平台，独立账户 |
| `kimi-search/kimi-k3` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | Responses API，当前官方仅列 K3 |

```rust
use lingxi_agent_api::protocol::{CompletionRequest, ConversationMessage, WebSearchConfig};
let mut request: CompletionRequest = serde_json::from_value(serde_json::json!({
    "model": "kimi-search/kimi-k3",
    "messages": []
})).unwrap();
request.messages.push(ConversationMessage::user_text("搜索今天的科技新闻并注明来源"));
request.web_search = Some(WebSearchConfig::default());
```

DeepSeek 官方 [Claude Code 接入文档](https://api-docs.deepseek.com/zh-cn/quick_start/agent_integrations/claude_code/) 声明其 Anthropic 端点支持原生搜索，而 [Responses 兼容表](https://api-docs.deepseek.com/guides/responses_api/) 明确将 `web_search` 列为忽略。故此处只在单独的 Anthropic 连接接入。发送 `web_search_20250305` 是依据 Claude 工具兼容性的推断：DeepSeek 未在文档中单独给出工具版本或高级选项，也尚未用真实凭证验证。原有 `deepseek` Chat 连接继续拒绝统一搜索选项。DeepSeek 搜索连接允许重放无签名 thinking 块，其他 Anthropic 连接仍要求原有签名。

GLM 使用 `enable: true`、`search_result: true`，国内默认引擎 `search_pro`，Z.AI 默认 `search-prime`。自定义连接可用 `extra.web_search_engine` 指定其账户可用的引擎。返回的 `web_search` 原始条目保存在 metadata，`link/title` 转为来源列表，`refer`、摘要及发布日期均保留；流式的无 choices 搜索帧也会保留。`glm-coding` 未声明该能力，因为 Coding Plan 的 MCP 搜索不是同一个聊天 API。

Kimi 使用[当前 Responses API](https://platform.kimi.com/docs/api/responses)的服务端搜索，不需要客户端执行或回传搜索工具。`include: ["web_search_call.action.sources"]` 请求原生来源；来源映射为 `citations`，完整搜索动作仍在 metadata 中。搜索模式不接受 `temperature` 或统一的 thinking token budget。旧的 `$web_search` builtin 接口已进入弃用周期，未将其接到 `kimi` Chat 或 `kimi-code` 订阅连接上。

新连接采用独立的 connection group，避免把普通聊天、订阅或另一地域账户作为搜索请求的自动备用连接。DeepSeek 与 Kimi 的 token 价格沿用仓库原有目录快照；按量计费的 `glm` 没有经过核实的美元单价，因此 `estimate_cost()` 返回 `None`，不会将 Coding Plan 的零价格误用于开放平台。搜索费仍不在 token 估算中。

## 搜索结果与引用

普通响应的 `response.web_search: Option<WebSearchResult>` 包含 `citations`（URL 和可选标题）及 `metadata`（原生搜索元数据）。没有搜索元数据时为 `None`。文本仍通过 `response.message.text()` 获取。

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

这是字段结构示意，`metadata` 内容随 provider 和响应格式而变；该例展示 GLM 的 `web_search` 字段，不是可直接当作固定格式解析的合同。所有来源的标准字段只承诺 `url` 和可选 `title`。

流式响应增加 `StreamEvent::WebSearch { result }`，表示该帧收到的搜索记录；不是需要执行的 `ToolCallDelta`。保留这些事件，不能只消费 `TextDelta`。各帧的 metadata 不是统一的累计对象；Gemini 的 grounding 数据按 provider 每帧返回的快照保留。展示来源时可按 URL 去重，但不要丢掉原生引用位置和关联信息。

原生元数据保留 OpenAI 的 annotation、Claude 的引用和搜索结果（包括搜索失败详情），以及 Gemini 的 grounding supports、查询和 Search Suggestions。引用的位置单位与块索引由各 provider 定义，不统一转换为 Rust 字节偏移。面向用户展示答案时，应利用这些字段展示可点击引用；Gemini 还要求展示返回的 Search Suggestions。客户端不生成 HTML 界面。

Claude 的普通响应把需要原样重放的搜索块和带搜索引用的文本保存为 `ContentBlock::ProviderContent`。将整个 `response.message` 加入下一轮 `request.messages`，即可保留 `encrypted_content`、`encrypted_index` 和搜索调用关系。`message.text()` 也会包含这些原生文本块，而 `message.tool_uses()` 不会把服务端搜索当成本地工具。

`StopReason::Other("pause_turn")` 表示 Claude 暂停了服务端循环；调用方可原样追加该 assistant 消息，再次请求继续。客户端不自动增加付费请求。流式调用方需要按块索引重建文本及原生搜索内容后才能重放；仅拼接文本会丢失加密上下文。原生内容重放到其他协议会被拒绝。

搜索失败也可能随 HTTP 200 返回，例如 Claude 的 `max_uses_exceeded`；此时搜索错误保留在 metadata 中，不会把失败伪装成成功引用。调用方应检查搜索记录，决定是否重试或告知用户。

## 费用与兼容性

搜索可能单独计费；现有 `estimate_cost()` 仅估算 token 费用，不包含搜索调用费。不要把它当作启用搜索后的总账单。保留 provider 返回的搜索用量元数据和已报告费用，以 provider 账单为准。

旧 JSON 请求/响应仍可反序列化，新增可选字段默认缺省。Rust 结构体字面量需补 `web_search: None`；穷尽匹配 `StreamEvent` / `ContentBlock` 的下游代码需处理新增变体。

## 官方文档

本实现依据以下官方接口文档，采用 Claude 基础搜索版本，以避免强制引入动态过滤及代码执行：

- [OpenAI Web search](https://developers.openai.com/api/docs/guides/tools-web-search)
- [Anthropic Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
- [Gemini Grounding with Google Search（generateContent）](https://ai.google.dev/gemini-api/docs/generate-content/google-search)
- [xAI Web Search](https://docs.x.ai/developers/tools/web-search)
- [DeepSeek Anthropic API](https://api-docs.deepseek.com/guides/anthropic_api/)
- [GLM 联网搜索](https://docs.bigmodel.cn/cn/guide/tools/web-search)
- [Z.AI Web Search in Chat](https://docs.z.ai/guides/tools/web-search)
- [Kimi Responses API](https://platform.kimi.com/docs/api/responses)
- [OpenRouter Web Search Server Tool](https://openrouter.ai/docs/guides/features/server-tools/web-search)

验证使用官方格式的本地 fixture 和模拟传输，不依赖真实 API key；不代表已验证账户权限、模型可用性或实际收费。
