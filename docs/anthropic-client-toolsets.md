# Anthropic Browser / Computer 客户端工具集

通过独立的 `ChatRequest.anthropic_client_toolsets` 声明稳定的 `browser_toolset_20260801` / `computer_toolset_20260801`。它们是客户端工具：LLM 返回调用，宿主负责权限、执行与结果。它们不属于 `hosted_tools`，客户端也不会打开浏览器或操纵桌面。

```rust
use lingxi_llm_client::protocol::{
    AnthropicBrowserMember, AnthropicBrowserToolsetConfig, AnthropicClientToolConfig,
    AnthropicClientToolset, ChatRequest,
};
# fn configure(request: &mut ChatRequest) {
let mut browser = AnthropicBrowserToolsetConfig::default();
browser.configs.insert(AnthropicBrowserMember::JavascriptExec,
    AnthropicClientToolConfig { enabled: Some(true), defer_loading: None });
request.anthropic_client_toolsets.push(AnthropicClientToolset::Browser(browser));
# }
```

工具条目没有 `name`，成员配置只接受 `enabled` / `defer_loading`。Browser 共 31 个成员，`javascript_exec`、`file_upload`、`read_console`、`read_network` 默认关闭，其他默认开启；Computer 的 17 个成员默认全部开启。省略配置保留提供方默认，不展开默认值。

两个工具集可以共存；允许自定义工具与成员同名。宿主必须根据 `(toolset_name, name)` 派发；`ToolUse`、`ToolResult`、`StreamEvent::ToolCallDelta` 均保留可选 `toolset_name`，`ConversationMessage::tool_uses()` 也返回该字段。结果必须回传调用的相同命名空间，普通工具的值为 `None`。Rust 字面量需要新增字段，不提供旧 API 兼容分支。

```rust
use lingxi_llm_client::protocol::{ContentBlock, ConversationMessage};
# fn results(message: &ConversationMessage) -> Vec<ContentBlock> {
message.content.iter().filter_map(|block| {
    let ContentBlock::ToolUse { id, toolset_name, name, .. } = block else { return None };
    // Dispatch using both toolset_name and name in the host application.
    Some(ContentBlock::ToolResult {
        tool_use_id: id.clone(),
        toolset_name: toolset_name.clone(),
        content: format!("Host execution unavailable for {toolset_name:?}/{name}"),
        is_error: true,
        blocks: None,
    })
}).collect()
# }
```

上述示例返回错误而不执行操作。成功的 `new_tab`、`switch_tab`、`close_tab`、`list_tabs` 返回恰好一个原生 `browser_state`，其他 Browser 成员允许 text/image 及最多一个状态块；错误结果不能附带状态。状态通过现有 `ToolResult.blocks` 保留，不把提供方渲染的状态文本伪装为客户端生成内容。Computer 结果只使用 text/image。

`allowed_callers` 省略或显式 `[Direct]`；不支持程序化 caller、strict、input_examples 或工具集顶层 defer_loading。所有启用成员必须具有相同 defer_loading，全部延迟时需要同请求的工具搜索，且不能在该工具集设置缓存断点。工具集缓存标记共享全请求四断点和 TTL 顺序检查。禁止全禁用、未知成员、重复工具集、同名 browser/computer 自定义工具、指向工具集或成员的强制 tool_choice，以及旧的 fine-grained streaming beta。

本实现支持第一方 Anthropic Messages 和 Vertex Claude 的已确认模型：Fable 5/5.1、Mythos 5/5.1、Opus 4.8/5/5.5、Sonnet 5，无需 beta。Vertex 使用相同的稳定工具集 wire ID 与命名空间规则；Foundry 和 Bedrock 的稳定工具集不在此接口支持范围。其他 codec 显式拒绝声明与历史命名空间，避免丢失信息。本地 token 估算将服务端定义的成员 schema 标为未计入，不能视为零成本。

验证仅使用离线/mock，不代表真实账户或宿主浏览器/桌面验收。官方依据：[工具参考](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)、[Browser](https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-use-tool)、[Computer](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)。

成员输出约定由宿主执行器负责：成功的 screenshot/zoom 应返回 image，其他非标签管理成员应返回 text，可附带额外图片。官方说明这些成员级输出约定不由 API 强制校验，因此客户端不将它们当作硬性协议限制；上面的原生块类型、标签管理及 browser_state 结构约束仍会校验。下载报告的 size_bytes 接受 null 或非负整数。
