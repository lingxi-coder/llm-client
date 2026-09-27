# Anthropic 程序化工具调用

[English](anthropic-programmatic-tools.en.md)

程序化工具调用允许 Anthropic Code Execution 容器内的代码调用当前请求中声明的函数工具。客户端负责编码请求、呈现调用方元数据，并保留原生状态以供回放。工具权限、实际执行和结果生成仍由宿主负责。

## 请求

在 `ChatRequest.tools` 的每个函数声明中设置 `allowed_callers`，并在同一请求中添加 `HostedTool::AnthropicCodeExecution`：

```rust
use lingxi_llm_client::protocol::{
    AnthropicCodeExecutionConfig, AnthropicToolCaller, ChatRequest, HostedTool, ToolSpec,
};
use serde_json::json;

let mut request: ChatRequest = serde_json::from_value(json!({
    "model": "claude-opus-5-5",
    "messages": [{"role":"user","content":[{"type":"text","text":"查找近期记录。"}]}]
})).unwrap();
request.hosted_tools.push(HostedTool::AnthropicCodeExecution(
    AnthropicCodeExecutionConfig::default(),
));
request.tools.push(ToolSpec {
    name: "lookup".into(),
    description: "查询记录".into(),
    input_schema: json!({
        "type": "object",
        "properties": { "query": { "type": "string" } },
        "required": ["query"]
    }),
    strict: false,
    defer_loading: false,
    allowed_callers: vec![AnthropicToolCaller::CodeExecution20260120],
});
```

`allowed_callers` 接受 `Direct`、`CodeExecution20260120` 和 `CodeExecution20260521`。Anthropic 的请求契约将两个 Code Execution 版本视为等价。省略该字段或提供空列表时，工具仍按通常方式由模型直接调用；同时列出 `Direct` 和 Code Execution caller 时，两种调用方式都可用。该字段支持第一方 Anthropic Messages 路由，也支持显式标记为 Anthropic hosting 且底层模型支持 PTC 的 Microsoft Foundry deployment。Azure-hosted Foundry 会被拒绝。Foundry body 的 `model` 仍使用自定义 deployment 名称，兼容性检查使用 `FoundryDeployment.model_id`。

客户端会在发送前检查已公开的限制：同一请求必须包含 Code Execution；所选模型必须支持程序化工具调用；程序化调用不支持严格工具和含递归本地 `$ref` 的 schema；`tool_choice` 不能强制选择仅允许程序化调用的工具；profile 级 `disable_parallel_tool_use: true` 也不能与该功能组合。Anthropic 文档列出的模型为 Fable 5/5.1、Mythos 5/5.1、Opus 4.5/4.6/4.7/4.8/5/5.5，以及 Sonnet 4.5/4.6/5。Haiku 4.5 和 Mythos Preview 接受 Code Execution，但不支持程序化工具调用。Foundry 还要求所选模型 ID 与 deployment 的 hosting 组合均在官方支持范围内。

`allowed_callers` 用于引导模型行为，不构成授权边界。宿主仍须执行自己的工具权限检查；不能仅因提供方使用了 Code Execution 就运行不可信命令。

## 响应与续传

在非流式响应中，Anthropic 的 `caller` 对象保存在 `ContentBlock::ToolUse.caller`。在流式响应中，可从 `StreamEvent::ToolCallDelta.caller` 读取该对象。客户端将它保留为原始 JSON，因此 `type`、`tool_id` 和未来新增字段都可以经由序列化与回放完整保留。程序化 caller 的 `tool_id` 指向前面的原生 `server_tool_use` block；该服务端工具 block 保持为 `ProviderContent`，不会重复创建宿主工具调用。

恢复尚未完成的程序化调用时，应回放 assistant 消息、返回匹配的 `ToolResult`、重新发送原始 `ToolSpec` 定义，并提供响应中的带作用域 Code Execution 容器引用。客户端会检查 caller 元数据是否指向此前的服务端工具 block，以及工具定义是否允许 Code Execution caller。客户端不会执行函数或代码、决定权限，也不负责保存对话历史。

已知 caller 格式不合法时，客户端会在发送前报错，同时仍在已解码响应中保留原始字段。`direct` 与未知 caller 对象可在 Anthropic Messages 各线协议中原样回放，包括第一方 API、Bedrock、Vertex、Azure AI Foundry，以及 OpenRouter 等兼容网关 profile。已识别的程序化工具 caller 仅在第一方 API，或底层模型身份已验证的 Anthropic-hosted Foundry deployment 上接受；这不代表 Azure-hosted Foundry、Bedrock、Vertex 或网关支持 PTC。OpenAI、Gemini 等其他协议 codec 会拒绝 caller 元数据，不会静默丢弃字段。

续传时，Foundry 容器引用必须绑定到同一个 Foundry resource、账号标识、deployment 和底层 model。Foundry Code Execution 文件使用独立的 Foundry Files service，并绑定 resource 和 account，而不绑定 container、deployment 或 model。自定义 Skills 可通过独立 Skills service 上传和管理；其 Foundry 引用也绑定 resource 和 account。Foundry 不支持下载 Skill version content。不要把第一方 Anthropic file reference 发到 Foundry。

容器作用域、上传文件、输出文件和失败处理见 [Code Execution](anthropic-code-execution.md)。当前实现仅通过 mock wire 测试；尚未验证真实账户可用性或 Anthropic 端的托管执行结果。

参考：Anthropic [程序化工具调用指南](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)、[工具参考](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)、[Code Execution 工具](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)和[Claude on Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)。
