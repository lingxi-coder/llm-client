# Vertex Claude 的工具与中途控制

`VertexClaudeCodec` 使用显式 `vertex_claude` 协议，不依赖自定义 provider_id 的拼写。模型保留在 publisher URL 中，body 使用 `anthropic_version: "vertex-2023-10-16"`；流式路径为 `streamRawPredict`，普通请求为 `rawPredict`。已有默认版本覆盖行为保持不变。Google 认证和项目/地区选择由调用方配置。

以下为纯编码示例，`auth: none` 不发送请求；实际调用需通过调用方的认证配置/Authenticator 提供 Google 凭证。

```rust
use lingxi_llm_client::{CodecContext, EncodeRequest, RequestMode, VertexClaudeCodec, WireCodec};
use lingxi_llm_client::protocol::{
    AnthropicToolSearchConfig, AnthropicToolSearchStrategy, ChatRequest, HostedTool, ProviderProfile,
};
use serde_json::json;
let profile: ProviderProfile = serde_json::from_value(json!({
    "provider_id": "google-vertex", "profile_name": "vertex",
    "protocol": "vertex_claude", "auth": "none",
    "base_url": "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/locations/us-central1",
    "models": [{"display_model":"claude-opus-5-5", "request_model":"claude-opus-5-5", "billing_model":"claude-opus-5-5"}]
})).unwrap();
let mut request: ChatRequest = serde_json::from_value(json!({
    "model":"claude-opus-5-5",
    "messages":[{"role":"user","content":[{"type":"text","text":"Find the right tool"}]}]
})).unwrap();
request.hosted_tools.push(HostedTool::AnthropicToolSearch(AnthropicToolSearchConfig {
    strategy: AnthropicToolSearchStrategy::Regex,
}));
let context = CodecContext::new(&profile, "claude-opus-5-5", RequestMode::Complete);
let wire = VertexClaudeCodec.encode_request(EncodeRequest::new(&request), &context).unwrap();
assert!(wire.url.ends_with("/publishers/anthropic/models/claude-opus-5-5:rawPredict"));
```

## Tool Search

Regex/BM25 使用原生稳定工具版本，无需新 beta。Claude API 和 Vertex 根据当前官方表校验模型；Vertex 的旧款模型使用公开的 `@YYYYMMDD` ID，不能用 Claude API 日期形式或显示名替换。Claude API 另外接受文档确认的三个 4.5 短别名。Bedrock InvokeModel 的不透明模型 ID/ARN 仍由提供方确认背后的模型，不在这里猜测。所有地区/账户可用性仍需真实环境验收。

## 中途 System 与工具引用

System 和 clear_at 支持 Fable 5/5.1、Mythos 5/5.1、Opus 4.8/5/5.5；逐消息 effort 支持其中 Fable 5.1、Mythos 5.1、Opus 5/5.5。使用精确 wire ID，不从任意后缀推断模型。消息组的位置、单轮提醒与缓存组合限制同第一方协议。

下面的 lookup 必须已经在 request.tools 中声明，消息必须追加在合法的 user 或服务端结果位置。Vertex 的工具引用增删使用 `mid-conversation-tool-changes-2026-07-01`；第一方内联定义的 beta 不自动用于 Vertex。内联定义和 MCP 添加仍拒绝。

```rust
use lingxi_llm_client::protocol::{
    AnthropicToolChange, AnthropicToolReference, ChatRequest, ConversationMessage, MessageRole,
};
# fn withdraw_declared_tool(request: &mut ChatRequest) {
request.messages.push(ConversationMessage {
    role: MessageRole::System,
    content: vec![AnthropicToolChange::remove(AnthropicToolReference::tool("lookup")).into_content_block()],
    anthropic: None,
});
# }
```

clear_at 与逐消息 effort 各自带独立 beta。客户端保留原始历史，effort 在下一条 user 消息时生效并反映到请求报告；不替宿主删除提醒或恢复会话。预检同时覆盖直接 codec 与高层客户端，错误组合在读取附件前拒绝。

## 稳定客户端工具集

Browser/Computer 的稳定 20260801 工具集在 Vertex 使用与第一方相同的声明、模型限制、缓存顺序和 toolset_name 往返。参见[客户端工具集](anthropic-client-toolsets.md)。执行、权限与浏览器/桌面状态由宿主负责。

当前 Google Cloud 平台指南与工具指南均列出 Browser/Computer 支持，更新了早期审查记录中的冲突结论。Web Fetch、Code Execution、MCP 等第一方能力仍使用各自平台限制，不因复用消息编码器而自动开放。Foundry 的托管类型/自定义部署模型身份仍需独立适配。

本轮验证为 mock/离线协议检查，不是 Google 真实账户验收。来源：[Google Cloud 平台](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai)、[Tool Search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)、[中途控制](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)、[官方 Vertex SDK](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/lib/vertex/_client.py)。
