# Anthropic Messages Web Fetch

`AnthropicHostedTool::WebFetch` 在第一方 Anthropic Messages 或明确标识的 Foundry Messages profile 上声明由提供方执行的抓取工具。Foundry 必须在对应 `ModelProfile` 行提供 hosting 和底层 Claude `model_id`；请求仍使用部署名。客户端不会自行下载 URL、生成宿主工具调用或自动重试结果不确定的执行；返回的原生块应保留用于续接。

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicWebFetchConfig, AnthropicWebFetchVersion};
use lingxi_llm_client::protocol::{ChatRequest, };
# fn configure(request: &mut ChatRequest) {
request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(AnthropicWebFetchConfig {
    version: AnthropicWebFetchVersion::V20260318,
    max_uses: Some(3),
    allowed_domains: vec!["example.com".into()],
    citations: Some(true),
    max_content_tokens: Some(4096),
    ..Default::default()
}).into());
# }
```

四个版本分别选择不同能力，不进行自动降级：

| 版本 | 控制 |
| --- | --- |
| `V20250910` | 基础抓取 |
| `V20260209` | 增加提供方动态内容过滤 |
| `V20260309` | 再增加 `use_cache` |
| `V20260318`（默认） | 再增加 `response_inclusion`：`Full` / `Excluded` |

动态过滤会由提供方隐式启用 `Code Execution`。Foundry Azure 托管仅接受 `V20250910` 基础版。Foundry Anthropic 托管可使用所有版本，但动态过滤只对当前文档列出的 Claude 4.6+ 与 Mythos Preview 底层模型开放；codec 用明确的 Foundry 身份核验，不要求另行声明 `Code Execution`。 Azure hosting 不提供 `Code Execution` / `PTC`，因此基础 Fetch 也只接受 `direct` caller。

对支持动态过滤的新版本，显式设置 `allowed_callers` 为 `Direct` 会关闭动态过滤，因此不适用动态过滤的模型限制；Foundry Azure 仍只接受 `V20250910`，Anthropic 托管的最终模型接受情况仍由服务端确认。

全部版本提供 `allowed_callers`、`strict`、`defer_loading`、五分钟/一小时 `cache_control` 和 `url_sources`。Fetch 专用 caller 枚举包含 `Direct` 及官方列出的 Code Execution caller 版本，不改变普通客户端函数的 PTC 契约。延迟工具必须有可用的启用方式，且不能带缓存断点。缓存标记共享四断点上限和 TTL 顺序；strict Fetch 计入二十个 strict 工具上限。

若要在中途 system 消息里按值加入 Web Fetch，请在此配置设置 `inline_definition: true`，并提供与该配置生成的 `tool_value()` 完全一致的 `tool_addition`。目前仅第一方 Claude API 支持此位置。类型化配置仍控制版本、模型、托管方、来源和重试校验，codec 只省略顶层 Web Fetch 项。如果配置了 `cache_control`，内联定义必须包含对应原生标记；该标记按所在消息位置计数，不会再作为顶层工具的合成标记重复计数。具体 API 版本是否接受该服务端工具按值添加，最终由服务端确认。

URL 来源策略保留省略时的默认值及显式空 `Only` / `Except` 列表：

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicFetchToolReference, AnthropicFetchToolResultsSources, AnthropicFetchUrlSources, AnthropicFetchUserInputSources};
let sources = AnthropicFetchUrlSources {
    user_input: Some(AnthropicFetchUserInputSources::All),
    client_tool_results: Some(AnthropicFetchToolResultsSources::Only {
        tools: vec![AnthropicFetchToolReference::new("lookup")],
    }),
    server_tool_results: Some(AnthropicFetchToolResultsSources::None),
};
// 赋给 config.url_sources，同时在 request.tools 中声明调用方工具 lookup。
```

来源引用必须对应当前请求顶层声明的工具；不猜测隐式 Code Execution 或 MCP 工具名。URL 是否满足提供方的来源规则仍由 Anthropic 判断，来源策略不会触发客户端本地抓取。

允许与禁止域名列表互斥，拒绝 scheme、空白和主机名通配符。有效的路径模式原样发送，但 Web Fetch 不匹配路径过滤条目，应使用主机名实现有效过滤。不为 `max_uses` / `max_content_tokens` 编造未公开的正数下界。

引用需显式启用，不能与 JSON-schema 输出组合；回放的抓取文档开启引用时也会检查。成功/错误结果和流式输入以 `ProviderContent` 保留，HTTP 200 中的抓取结果错误不转成传输失败。`Usage.server_tool_usage.web_fetch_requests` 区分明确报告的零与缺失；`anthropic_usage` 及原生流事件保留原始用量，不根据次数推算未报告费用。

Foundry 路由与模型身份说明见[Foundry 指南](anthropic-foundry.md)。本轮仅离线/mock 验证，不代表真实账户验收。依据：[Web Fetch 指南](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)、[工具参考](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)、[Microsoft Foundry 上的 Claude](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)、[官方请求 schema](https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/src/resources/messages/messages.ts)。

[Server tools: callers, dynamic filtering and domain rules](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools).
