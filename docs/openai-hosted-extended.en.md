# OpenAI Responses: web search page-open actions and remote MCP

This adapter implements the OpenAI Responses hosted `web_search` and remote MCP wire. Current OpenAI docs do not define a separate `web_fetch` tool type: page retrieval is an `open_page` action inside `web_search_call.action`. The codec retains full native `web_search_call` output and citation annotations, and requests `web_search_call.action.sources` so the host can inspect source records and page-open actions.

For deferred function or MCP discovery, see [OpenAI Responses Tool Search](openai-tool-search.en.md).

The profile must declare:

```toml
protocol = "open_ai_responses"

[extra]
web_search = "openai_responses"
remote_mcp = "openai_responses"
```

## Hosted web search and page opening

Use the existing `HostedTool::WebSearch`. OpenAI's wire uses `type: "web_search"`; `allowed_domains` and `blocked_domains` map together to `filters` using the official REST field names. Search and page-open actions execute on OpenAI's side and do not produce a host-executable `ToolUse`.

```rust,no_run
use lingxi_llm_client::protocol::{HostedTool, WebSearchConfig};

let web = HostedTool::WebSearch(WebSearchConfig {
    allowed_domains: vec!["openai.com".into()],
    blocked_domains: vec!["example.invalid".into()],
    ..Default::default()
});
```

`max_uses` is an Anthropic-only control, so the OpenAI Responses codec rejects it before sending. The native `web_search_call` output contains the action (`search`, `open_page`, or `find_in_page`); message `url_citation` annotations are also retained.

## Remote MCP

`RemoteMcpConfig` contains only non-secret server settings. OAuth tokens are injected for the current request through `RequestOptions.mcp_authorizations`, keyed by `server_label`; they do not enter serializable `ChatRequest` values or history. The caller must provide a fresh authorization token for every Responses creation request, including a request that continues after an approval decision. The client does not connect to the MCP server or execute MCP tools on the host.

```rust,no_run
use lingxi_llm_client::{
    protocol::{HostedTool, McpApprovalPolicy, RemoteMcpConfig, Secret},
    RequestOptions,
};
use std::collections::BTreeMap;

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let remote = RemoteMcpConfig::new("docs", "https://mcp.example.test/mcp")?
    .with_description("Read-only documentation")
    .with_allowed_tools(["search", "fetch"])?
    .with_require_approval(McpApprovalPolicy::Always);
let hosted_tool = HostedTool::RemoteMcp(remote);

let mut options = RequestOptions::default();
options.mcp_authorizations = BTreeMap::from([(
    "docs".into(),
    Secret::new("Bearer caller-managed-oauth-token".into()),
)]);
# Ok(())
# }
```

`RemoteMcpConfig` encodes `type`, `server_label`, `server_url`, optional `server_description`, optional `allowed_tools`, and optional `require_approval` (`always` or `never`). The `server_url` must use HTTPS. Remote MCP is gated by an explicit profile declaration, `extra.remote_mcp = "openai_responses"`, and the OpenAI Responses codec.

OpenAI output items `mcp_list_tools`, `mcp_call`, and `mcp_approval_request` are preserved as `ContentBlock::ProviderContent` with `ProtocolFamily::OpenAiResponses`. They never appear in `ConversationMessage::tool_uses()`. An approval request sets stop reason `Other("requires_action")`; the host can inspect its original server label, tool name, arguments, and item ID using `McpApprovalRequest::from_native_item`.

After the host decides, `McpApprovalRequest::respond(approve).into_assistant_message()` creates the native `mcp_approval_response` input item with `approval_request_id` and `approve`. Add it to the next request and pass the prior response continuation. The raw item replays only on OpenAI Responses; the codec rejects it on other protocols. Approval belongs to the caller: this client neither decides nor executes tools.

Official references: [OpenAI Web Search guide](https://developers.openai.com/api/docs/guides/tools-web-search) and [OpenAI MCP servers guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp).
