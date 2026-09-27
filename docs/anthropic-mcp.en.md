# Anthropic Messages MCP Connector

HostedTool::AnthropicMcp configures one remote MCP server and its matching Anthropic mcp_toolset. The first-party Messages API sends the mcp-client-2026-09-15 beta and supports the newer list-pinning and inline-toolset controls below. Both Microsoft Foundry hosting options, Azure and Anthropic, use the mcp-client-2025-11-20 beta for the base connector. Foundry MCP does not require a Foundry model identity or model allowlist.

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

Each server URL must use HTTPS, server names must be unique, and Anthropic accepts at most 20 servers. A server name is paired with exactly one toolset by this config. The caller owns OAuth login and token refresh. Tokens belong in request-scoped `RequestOptions::mcp_authorizations`, keyed by the server `name`; they are not part of `ChatRequest`, persisted history, or provider profile configuration. The client injects a supplied `Secret<String>` only into that request’s matching `mcp_servers[].authorization_token` field.

Use `default_config` for defaults and `with_tool_config` for per-tool `enabled` or `defer_loading` overrides. Names in `configs` are forwarded without a local server lookup because Anthropic allows MCP servers to expose dynamic tool names. A deferred MCP tool requires an Anthropic hosted tool-search tool on the same request.

The mcp-client-2026-09-15 beta adds pinned tool lists and inline operations for the first-party Messages connector. Use with_tools there to send the list returned in an mcp_tool_listing block. The current public Foundry documentation establishes only the mcp-client-2025-11-20 base connector, not availability of those newer beta operations. This client therefore rejects pinned lists, replayed mcp_tool_listing blocks, and inline MCP additions on Foundry.

```rust
use lingxi_llm_client::protocol::{AnthropicMcpConfig, AnthropicMcpTool};
use serde_json::json;

# fn pinned() -> Result<AnthropicMcpConfig, Box<dyn std::error::Error>> {
let connector = AnthropicMcpConfig::new("docs", "https://mcp.example.com/sse")?
    .with_tools([AnthropicMcpTool {
        name: "search".into(),
        description: Some("Search the documentation".into()),
        input_schema: json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }),
    }])?;
# Ok(connector)
# }
```

Leaving `tools` unset lets Anthropic fetch the list. Passing an empty iterator explicitly pins an empty list. When a response contains `mcp_tool_listing`, replay the assistant content unchanged and keep the same MCP server/toolset configuration on the next request. MCP tool uses, results, and listings remain Anthropic `ProviderContent`; the host does not receive duplicate client-side `ToolUse` calls and does not execute these hosted tools.

MCP toolset `cache_control` supports the documented five-minute and one-hour lifetimes. Its marker counts with the request’s other Anthropic prompt-cache breakpoints and follows tool breakpoints, before system and message breakpoints. The request-wide limit and TTL ordering checks still apply.

The connector only provides remote tool calls over Streamable HTTP or SSE. It does not expose prompts, resources, or local STDIO servers. Mid-conversation server additions and inline toolsets use the newer beta on the first-party Anthropic Messages route only; the Foundry route rejects those newer features. See [mid-conversation instructions and tools](anthropic-conversation.en.md). The codec selects the beta for its route without rewriting history.

Requests with a typed connector or replayed native MCP blocks are pinned to one attempt. If the transport outcome is uncertain, the client returns the error without automatic retry or failover; this avoids sending a potentially repeated remote action. The caller decides how to resume.

The current connector guide uses Claude Opus 5.5 in its examples but does not state an exclusive model allowlist. The client therefore does not infer one from the example; actual model availability remains subject to Anthropic’s API.

Raw body.mcp_servers, mcp_toolset, and authorization-token fields are rejected on first-party Anthropic Messages profiles; use typed configuration for that route. Foundry uses the typed base configuration here. Compatible gateways can still replay native Anthropic MCP history as opaque provider content, but the client does not enable typed MCP configuration or add the first-party beta header for them.

The connector uses no OpenAI approval policy. Anthropic MCP calls execute through Anthropic’s server-side connector, so the OpenAI `require_approval` contract does not apply.

References: [Anthropic MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector), [Messages API reference](https://platform.claude.com/docs/en/api/beta/messages/create), [tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching), and [mid-conversation tool changes](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages).
