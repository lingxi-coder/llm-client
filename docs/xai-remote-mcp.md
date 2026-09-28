# xAI Remote MCP（Responses）

此适配器在 xAI 的 OpenAI 兼容 Responses 路由上配置由 xAI 托管调用的 MCP 服务。当前 xAI Responses 专用配置为 `XaiHostedTool::RemoteMcp(XaiRemoteMcpConfig)`，只包含公开连接参数：`server_label`、HTTPS `server_url`、可选 `server_description` 和可选 `allowed_tools`。`allowed_tools` 为空表示不限制服务端公布的工具；非空时只开放列出的名称。

配置只接受启用 xAI Responses MCP 的官方配置档：`provider_id = "xai"`、`protocol = "open_ai_responses"`、`base_url = "https://api.x.ai/v1"`，以及 `extra.xai_remote_mcp = "xai_responses"`。它不会把既有 Grok Chat 路由切换到 Responses。编码后的 MCP 工具使用 xAI 文档示例确认的 `type: "mcp"`、`server_url`、`server_label`，并仅在配置了时附带 `server_description` 和 `allowed_tools`。

MCP 服务令牌通过每次请求的 `RequestOptions::mcp_authorizations` 按 `server_label` 传入。令牌只在发送前写入匹配的 Responses 工具对象的 `authorization` 字段；它不属于 `ChatRequest`，不会进入聊天历史或序列化请求配置。客户端会在处理附件或发送网络请求前校验提供商、协议、官方端点、功能标记和令牌标签。xAI 与 OpenAI MCP 配置不能在同一个请求中混用；带 MCP 工具的请求也不会跨配置档自动重试。

响应中的 `mcp_call` 及其他未知 MCP 输出项保留为 `ProviderContent` 原生数据。客户端不将它们转换成本地 `ToolUse`，不会执行服务器工具，也不会把 OpenAI 的 `mcp_approval_request` 语义套用到 xAI。xAI 文档明确表示其 Responses API 当前不支持 `require_approval` 和 `connector_id`，因此这两个字段不在此配置类型中；该类型拒绝包含这些未知字段的反序列化输入。其他任意 MCP `headers` 尚未建模。

官方的 [Remote MCP 功能指南](https://docs.x.ai/developers/tools/remote-mcp)列出 Responses API 支持并给出请求样例；[工具使用详情](https://docs.x.ai/developers/tools/tool-usage-details)将 Responses 输出项中的 `mcp_call` 标为由 xAI 服务器处理的 MCP 工具；[Streaming & Sync 指南](https://docs.x.ai/developers/tools/streaming-and-sync)说明 Responses 中 MCP 项会返回。通用 [Responses REST 参考](https://docs.x.ai/developers/rest-api-reference/inference/responses)目前仍只列出函数和网页搜索工具；这是与专用 MCP 指南的文档差异。本适配器以更具体的 MCP 功能指南为依据，不推测其未展示的其他字段。实际账户与服务端验收尚未执行。
