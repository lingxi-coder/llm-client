# Anthropic 中途系统指令与工具变更

第一方 Anthropic Messages 和 Vertex Claude codec 支持在会话历史中使用 `MessageRole::System`，限官方列出的 Fable 5/5.1、Mythos 5/5.1、Opus 4.8、Opus 5 和 Opus 5.5 wire model ID。不支持的模型及兼容网关会在本地拒绝；Vertex 只开放有当前官方证据的消息控制与工具引用，Foundry 和 Bedrock 的中途控制仍待独立适配。

连续 system 消息作为一个区段校验：前面必须是 user 消息（含工具结果），或以服务端工具结果结束的 assistant 消息；后面必须是 assistant 消息或数组结束。暂停的服务端工具轮次之后可以追加文本，但不能改变工具。初始系统指令仍放在 `ChatRequest.system`。

```rust
use lingxi_llm_client::protocol::{ChatRequest, ContentBlock, ConversationMessage, MessageRole};
# fn append(request: &mut ChatRequest) {
request.messages.push(ConversationMessage {
    native_options: Vec::new(),
    role: MessageRole::System,
    content: vec![ContentBlock::Text {
        text: "后续回答请明确注明单位。".into(),
        thought_signature: None,
    }],
});
# }
```

工具增删采用 system 消息中的 Anthropic `ProviderContent` block。codec 自动选择 `inline-tools-2026-09-15`；纯文本不需要 beta。自定义工具定义及引用保持原始 wire 结构。同类型工具可以用新定义更新，删除只能通过引用表达。

```rust
use lingxi_llm_client::protocol::{ContentBlock, ConversationMessage, MessageRole, ProtocolFamily};
use serde_json::json;
let message = ConversationMessage {
    native_options: Vec::new(),
    role: MessageRole::System,
    content: vec![ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: json!({
            "type": "tool_addition",
            "tool": {
                "type": "tool_definition",
                "definition": {
                    "name": "lookup",
                    "description": "查询标识符",
                    "input_schema": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"]
                    }
                }
            }
        }),
    }],
};
```

定义里的 `cache_control` 按消息位置计数。缓存标记只能放在 addition block 或 definition 其中一处；延迟加载的定义不能设置断点。四个断点上限和 TTL 顺序校验同样生效。客户端不重写历史，也不管理历史或缓存生命周期。

单轮 `clear_at` 和逐消息 effort 已提供，见下文。Anthropic 定义的服务端和客户端工具仍须走各自类型化配置，不能通过通用 `tool_definition` 绕过模型、路由和重试校验。Code Execution 和 Web Fetch 可在对应配置设置 `inline_definition: true` 后按值加入中途 system 消息。请求中必须有与类型化配置完全一致的 `tool_addition`；codec 继续执行模型和路由检查、固定自动重试路径，并且只省略该工具的顶层 `tools` 项。此位置目前仅支持第一方 Claude API。Tool Search 与完整的延迟目录仍放在顶层 `tools`。Claude API 支持用按名称引用的 `tool_addition` 在中途提供一个已知的顶层延迟工具。Tool Search 文档要求可搜索定义保留在顶层 `tools`，没有说明按值加入的内联定义也会进入搜索目录，因此本客户端不承诺这种发现行为。MCP 工具集也要求匹配的 `AnthropicMcpConfig`。工具执行、审批、历史保留和凭证刷新归宿主管理。本次验证采用离线和 mock，不代表真实账户验收。

依据：[Anthropic 中途 system 消息与工具变更](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)。

## 中途加入 MCP 服务器

在 `hosted_tools` 中保留服务器配置，标记工具集使用内联位置，并在工具开始可用时追加生成的 system 消息。服务器 URL 仍在 `mcp_servers`，工具集不会重复加入顶层 `tools`。授权继续使用以服务器名为键的 `RequestOptions.mcp_authorizations`。codec 自动发送两个必需 beta。

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicMcpConfig};
use lingxi_llm_client::protocol::{ChatRequest, };
# fn append(request: &mut ChatRequest) -> Result<(), Box<dyn std::error::Error>> {
let server = AnthropicMcpConfig::new("calendar", "https://mcp.example.com/calendar")?
    .with_inline_toolset(true);
request.messages.push(server.inline_tool_addition_message()?);
request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(server).into());
# Ok(())
# }
```

`AnthropicToolChange` 和 `AnthropicToolReference` 也可生成自定义定义或引用增删的原生 block/system 消息。内联自定义定义复用 Messages strict schema 子集、引用检查和请求级复杂度上限；重复的相同名称/schema 只计一次，变化后的 schema 仍计入同一请求。`allowed_callers` 接受 `direct`、`code_execution_20260120` 或 `code_execution_20260521`。程序化调用需要同一请求中的类型化 Code Execution 配置和受支持模型；递归 `$ref` 与 `strict: true` 不能用于程序化工具。历史调用按发生位置的工具定义校验，未完成的调用还要求当前历史末尾仍保留允许 Code Execution 的定义；删除或改成 direct-only 会在发送前失败。类型化 Code Execution 和 Web Fetch 只有在对应配置启用 `inline_definition`，且内联定义与配置完全一致时，才可按值放入中途消息。其他 Anthropic 服务端/客户端工具不能通过通用内联定义加入，以保留各功能的专属校验。渲染后的工具文本大小及远程发现的 MCP 工具数无法从本地完整预知，由服务端作最终校验。

依据：[strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)、[programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)、[tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)。

## 单轮提醒与逐消息 effort

使用 `ConversationMessage::with_anthropic_options()` 设置消息级控制。它被编码为消息上的 `clear_at` 与 `output_config.effort`，不是 content block。其他内置协议路由会拒绝这些控制。Rust 消息结构体字面量现在需包含 `native_options: Vec::new()`；普通消息优先使用构造方法。

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicClearAt, AnthropicMessageOptions};
use lingxi_llm_client::protocol::{ConversationMessage};
let reminder = ConversationMessage::system_text("请一起发起相互独立的读取操作。")
    .with_anthropic_options(AnthropicMessageOptions {
        clear_at: Some(AnthropicClearAt::NextUserMessage),
        effort: None,
    });
```

`NextUserMessage` 只允许一个或多个文本块，遵循普通 system 位置规则，不能同时设置 effort、工具变更或缓存断点。后续 user 消息（包括工具结果）使服务端停止渲染该提醒；历史中的原始消息仍须原样保留，客户端不会删除它。显式 `Never` 保持长期消息语义。显式设置任一值都会加入 clear-at beta；自动缓存仍交给 Anthropic 选择可用断点。

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicMessageEffort, AnthropicMessageOptions};
use lingxi_llm_client::protocol::{ConversationMessage};
let mut change = ConversationMessage::system_text("")
    .with_anthropic_options(AnthropicMessageOptions {
        clear_at: None,
        effort: Some(AnthropicMessageEffort::Low),
    });
change.content.clear();
```

只有 effort 且 content 为空的消息可以放在任意位置；连续 system 区段中一旦含有文本或工具变更，整个区段就按普通位置规则校验。逐消息 effort 限 Fable 5.1、Mythos 5.1、Opus 5.5 与 Opus 5，支持 low、medium、high、xhigh、max，自动加入 output-config beta。它从下一条 user 消息（包括工具结果）起生效，`InferenceReport.requested_effort` 按这个边界报告，不改写顶层设置。effort 不能与 `NextUserMessage` 组合。

依据：[逐消息 effort](https://platform.claude.com/docs/en/build-with-claude/effort#per-message-effort-beta)。同样适用于本库的 Vertex Claude 路由。

## Vertex 工具引用

Vertex 的工具增删只接受对顶层已声明工具的引用，并使用 `mid-conversation-tool-changes-2026-07-01`。第一方内联定义使用的 `inline-tools-2026-09-15` 不会自动发送到 Vertex，内联定义与内联 MCP 明确拒绝。消息选项的独立 beta 保持不变。详见[Vertex 指南](anthropic-vertex.md)。
