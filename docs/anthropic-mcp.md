# Anthropic Messages MCP 连接器

HostedTool::AnthropicMcp 用于配置一个远程 MCP 服务器及其 Anthropic mcp_toolset。第一方 Messages API 使用 mcp-client-2026-09-15 beta，并支持下文说明的新版本列表固定和内联工具集控制；Microsoft Foundry 的 Azure 与 Anthropic 两种托管方式使用 mcp-client-2025-11-20 beta 基础连接器。Foundry 的 MCP 基础配置不要求 Foundry 模型身份或模型白名单。

```rust
use lingxi_llm_client::{
    protocol::{
        AnthropicMcpConfig, AnthropicMcpToolConfig, ChatRequest, HostedTool, Secret,
    },
    RequestOptions,
};

# fn configure(request: &mut ChatRequest) -> Result<RequestOptions, Box<dyn std::error::Error>> {
let connector = AnthropicMcpConfig::new(
    "docs",
    "https://mcp.example.com/sse",
)?
.with_default_config(AnthropicMcpToolConfig {
    enabled: Some(true),
    defer_loading: Some(false),
});

request.hosted_tools.push(HostedTool::AnthropicMcp(connector));

let mut options = RequestOptions::default();
options.mcp_authorizations.insert(
    "docs".into(),
    Secret::new("access-token-from-your-OAuth-flow".into()),
);
# Ok(options)
# }
```

服务器 URL 必须使用 HTTPS，服务器名称必须唯一；Anthropic 每个请求最多接受 20 个服务器。此配置会让每个服务器名称对应且只对应一个 toolset。OAuth 登录和令牌刷新由调用方负责。令牌应放在按请求传入的 `RequestOptions::mcp_authorizations` 中，并以服务器 `name` 为键；它不会进入 `ChatRequest`、持久化历史或 provider profile。客户端只会将提供的 `Secret<String>` 注入本次请求中匹配的 `mcp_servers[].authorization_token`。

使用 `default_config` 设置默认值，使用 `with_tool_config` 设置单个工具的 `enabled` 或 `defer_loading` 覆盖项。由于 MCP 服务器可能动态改变工具名称，客户端会原样转发 `configs` 中的名称，不会先向服务器查询并据此拒绝未知名称。延迟加载的 MCP 工具必须与 Anthropic 托管工具搜索一起使用。

`mcp-client-2026-09-15` beta 支持固定服务器工具列表。可将响应 `mcp_tool_listing` 中的工具列表传给 `with_tools`：

```rust
use lingxi_llm_client::protocol::{AnthropicMcpConfig, AnthropicMcpTool};
use serde_json::json;

# fn pinned() -> Result<AnthropicMcpConfig, Box<dyn std::error::Error>> {
let connector = AnthropicMcpConfig::new("docs", "https://mcp.example.com/sse")?
    .with_tools([AnthropicMcpTool {
        name: "search".into(),
        description: Some("搜索文档".into()),
        input_schema: json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }),
    }])?;
# Ok(connector)
# }
```

不设置 `tools` 时，Anthropic 会查询服务器工具列表。传入空迭代器则会明确固定一个空列表。如果响应包含 `mcp_tool_listing`，下一轮应原样重放 assistant 消息，并继续提供相同的 MCP 服务器和 toolset 配置。MCP 工具调用、结果和列表都保留为 Anthropic `ProviderContent`；客户端不会把托管工具转换成重复的本地 `ToolUse`，也不会在宿主侧执行它们。

MCP toolset 的 `cache_control` 支持文档规定的 5 分钟和 1 小时 TTL。其断点会与请求中其他 Anthropic 提示缓存断点一起计数，顺序位于普通工具断点之后、system 和 message 断点之前。请求总断点数量及 TTL 顺序校验仍然生效。

此连接器只支持远程 Streamable HTTP 或 SSE 工具调用，不提供 prompts、resources 或本地 STDIO。中途添加服务器及内联工具集仅适用于第一方 Anthropic Messages 的新 beta；Foundry 路由会拒绝这些较新功能。详见[中途指令与工具变更](anthropic-conversation.md)。codec 会按路由选择相应 beta，不改写既有历史。

请求包含类型化 MCP 配置或重放原生 MCP 历史时，客户端会固定在单次请求尝试上。如果传输结果不确定，客户端会返回错误，不会自动重试或故障转移，以免重复执行可能已发生的远程操作。调用方可自行决定如何恢复。

当前连接器指南的示例使用 Claude Opus 5.5，但没有声明只允许该模型。客户端不会从单个示例推断模型白名单；具体可用性仍由 Anthropic API 决定。

第一方 Anthropic Messages profile 上的原始 body.mcp_servers、mcp_toolset 和授权令牌字段会被拒绝；第一方连接器应使用类型化配置。Foundry 使用此处的类型化基础配置。兼容网关仍可将原生 Anthropic MCP 历史作为不透明 provider 内容重放，但客户端不会为网关启用类型化 MCP 配置或添加第一方 beta header。

Anthropic MCP 连接器不使用 OpenAI approval policy。此类调用经 Anthropic 服务器端连接器执行，因此 OpenAI 的 `require_approval` 合约不适用。

参考：[Anthropic MCP 连接器](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)、[Messages API 参考](https://platform.claude.com/docs/en/api/beta/messages/create)、[工具调用与提示缓存](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)、[对话中的工具变更](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)。
