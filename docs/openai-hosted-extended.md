# OpenAI Responses：Web Search 的打开页面动作与 Remote MCP

此适配器封装 OpenAI Responses 的 hosted `web_search` 和 remote MCP wire。OpenAI 当前文档没有单独的 `web_fetch` tool type：页面读取是 `web_search_call.action` 中的 `open_page` 动作。适配器保留完整 `web_search_call` 输出和引用注释，并请求 `web_search_call.action.sources`，以便宿主检查来源和页面读取动作。

延迟加载函数或 MCP 工具请参阅 [OpenAI Responses Tool Search](openai-tool-search.md)。

profile 必须声明：

```toml
protocol = "open_ai_responses"

[extra]
web_search = "openai_responses"
remote_mcp = "openai_responses"
```

## Hosted Web Search / 页面读取

使用已有的 `HostedTool::WebSearch`。OpenAI wire 使用 `type: "web_search"`；`allowed_domains` 与 `blocked_domains` 同时映射到 `filters`，并采用官方 REST 字段名。Hosted 页面读取和搜索都是 OpenAI 执行的动作，不会产生需要宿主运行的 `ToolUse`。

```rust,no_run
use lingxi_llm_client::protocol::{HostedTool, WebSearchConfig};

let web = HostedTool::WebSearch(WebSearchConfig {
    allowed_domains: vec!["openai.com".into()],
    blocked_domains: vec!["example.invalid".into()],
    ..Default::default()
});
```

`max_uses` 是 Anthropic 专属控制，OpenAI Responses codec 会在发送前拒绝。响应中的 `web_search_call` 原生项含 action (`search`、`open_page` 或 `find_in_page`)；它和消息上的 `url_citation` annotation 都保存在输出中。

## Remote MCP

`RemoteMcpConfig` 只包含非秘密服务设置；OAuth token 通过 `RequestOptions.mcp_authorizations` 按 `server_label` 注入到本次 Responses 请求的 `authorization` 字段。它不会进入可序列化的 `ChatRequest` 或历史。调用方必须在每次 Responses 创建请求（包括继续审批后的请求）重新提供当前授权 token。客户端不连接 MCP 服务器，也不在宿主执行 MCP 工具。

```rust,no_run
use lingxi_llm_client::providers::openai::types::{McpApprovalPolicy, RemoteMcpConfig};
use lingxi_llm_client::{
    protocol::{HostedTool, Secret},
    RequestOptions,
};
use std::collections::BTreeMap;

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let remote = RemoteMcpConfig::new("docs", "https://mcp.example.test/mcp")?
    .with_description("Read-only documentation")
    .with_allowed_tools(["search", "fetch"])?
    .with_require_approval(McpApprovalPolicy::Always);
let hosted_tool: lingxi_llm_client::protocol::HostedTool = lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(remote).into();

let mut options = RequestOptions::default();
options.mcp_authorizations = BTreeMap::from([(
    "docs".into(),
    Secret::new("Bearer caller-managed-oauth-token".into()),
)]);
# Ok(())
# }
```

`RemoteMcpConfig` emits `type`, `server_label`, `server_url`, optional `server_description`, optional `allowed_tools`, and `require_approval` (`always` or `never`). The `server_url` must be HTTPS. Remote MCP is gated by the profile's explicit `extra.remote_mcp = "openai_responses"` declaration and the OpenAI Responses codec.

When OpenAI returns `mcp_list_tools`, `mcp_call`, or `mcp_approval_request`, each item is retained as `ContentBlock::ProviderContent` with `ProtocolFamily::OpenAiResponses`. They do not appear in `ConversationMessage::tool_uses()`. An approval request changes the stop reason to `Other("requires_action")`; the host decides whether to approve and may inspect the original server label, tool name, arguments and item ID through `McpApprovalRequest::from_native_item`.

After the host decides, `McpApprovalRequest::respond(approve).into_assistant_message()` creates the native `mcp_approval_response` input item (`approval_request_id` and `approve`). Add it to the next request and pass the prior response continuation. The raw item is replayed on the OpenAI Responses protocol only; it cannot be routed to another protocol. Approval is caller-owned: this client does not make the decision or execute tools.

Official references: [OpenAI Web Search guide](https://developers.openai.com/api/docs/guides/tools-web-search) and [OpenAI MCP servers guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp).
