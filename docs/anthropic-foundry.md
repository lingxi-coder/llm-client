# Microsoft Foundry 上的 Anthropic Claude

[English](anthropic-foundry.en.md)

Foundry 通过 Foundry 资源端点承载 Anthropic Messages wire。使用协议 `foundry_claude`，base URL 格式为 **https://<resource>.services.ai.azure.com/anthropic**。普通 Messages 请求无需额外的 Foundry 模型身份。Tool Search、Web Fetch、Code Execution 和 Programmatic Tool Calling 则需要：必须在准确的模型行上设置 `ModelProfile.foundry`，因为请求中的 model 是部署名，不能据此判断底层模型或 hosting。

| 字段 | 用途 |
| --- | --- |
| `request_model` | 发送在 Messages body 中的 Foundry 部署名，即 model 值 |
| `foundry.hosting` | 创建部署时选择的 hosting：**azure** 或 **anthropic** |
| `foundry.model_id` | 用于能力校验的精确 Claude 底层模型 ID |
| `billing_model` | 定价查找键；不会用于推断模型 |

不要从部署名、显示名、计费 ID、端点或 Azure 资源推断 hosting 或模型身份。若多个 profile 模型行具有相同部署名但身份信息不同，请用 `CodecContext::for_model` 选择目标行；仅从 wire model 无法区分时，身份敏感的托管工具会返回歧义错误。身份元数据不会发送到请求 body。

## Tool Search

Anthropic 托管 Tool Search 支持 Foundry 的两种 hosting，但底层模型必须同时出现在 Foundry 和 Tool Search 的文档兼容范围内。codec 使用两张表的交集，不根据模型系列前缀推断兼容性。

| Foundry hosting | 可用的底层模型 ID |
| --- | --- |
| Azure | `claude-opus-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-haiku-4-5` |
| Anthropic | `claude-fable-5-1`, `claude-mythos-5-1`, `claude-fable-5`, `claude-mythos-5`, `claude-opus-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5` |

4.5 Foundry 部署应使用文档中的不带日期后缀 ID，例如 `claude-opus-4-5`、`claude-sonnet-4-5` 和 `claude-haiku-4-5`。虽然 Foundry 提供 Sonnet 5，但它不在 Tool Search 的模型列表中。账户、订阅和部署的实际可用性仍由服务端决定。

请求里的 model 仍是部署名。下面的例子只做本地校验和请求编码，不会发起网络调用，也不配置凭证：

~~~rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicToolSearchConfig, AnthropicToolSearchStrategy};
use lingxi_llm_client::protocol::{ChatRequest, FoundryDeployment,
    FoundryHosting, ProviderProfile};
use lingxi_llm_client::{
    CodecContext, EncodeRequest, FoundryClaudeCodec, RequestMode, WireCodec,
};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "my-foundry-resource",
        "profile_name": "foundry-claude",
        "protocol": "foundry_claude",
        "base_url": "https://my-resource.services.ai.azure.com/anthropic",
        "auth": "none",
        "models": [{
            "display_model": "Claude Opus",
            "request_model": "prod-opus-55",
            "billing_model": "accounting-opus"
        }]
    }))?;
    profile.models[0].foundry = Some(FoundryDeployment {
        hosting: FoundryHosting::Anthropic,
        model_id: "claude-opus-5-5".into(),
    });

    let mut request: ChatRequest = serde_json::from_value(json!({
        "model": "prod-opus-55",
        "messages": [{"role":"user","content":[{"type":"text","text":"Find a calendar tool"}]}]
    }))?;
    request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
        AnthropicToolSearchConfig {
            strategy: AnthropicToolSearchStrategy::Bm25,
        },
    ).into());

    let context = CodecContext::new(&profile, "prod-opus-55", RequestMode::Complete);
    FoundryClaudeCodec.validate_request(&request, &context)?;
    let wire = FoundryClaudeCodec.encode_request(EncodeRequest::new(&request), &context)?;
    let body: Value = serde_json::from_slice(&wire.body)?;
    assert_eq!(body["model"], "prod-opus-55");
    assert_eq!(body["tools"][0]["type"], "tool_search_tool_bm25_20251119");
    Ok(())
}
~~~

## Web Fetch

类型化 Web Fetch 支持 Foundry endpoint，并要求对应模型行显式提供 Foundry hosting 与底层 model ID。Azure hosting 仅支持基础版 `web_fetch_20250910`，且只接受 direct caller；非 direct 的 Code Execution caller 会被拒绝，因为 Azure hosting 不提供 Code Execution / 程序化工具调用（PTC）。Anthropic hosting 支持 Web Fetch 的全部四个版本，见 [Web Fetch 指南](anthropic-web-fetch.md)；其中动态过滤仍要求底层模型在该功能的文档兼容范围内。对支持动态过滤的版本设置 direct-only caller 会关闭动态过滤。codec 会在发送前拒绝不支持的 hosting、版本、模型或 caller 组合。

## MCP connector

Foundry 的 Azure 与 Anthropic 两种 hosting 均支持使用 `mcp-client-2025-11-20` 的类型化远程 MCP 基础连接器。该基础连接不要求 ModelProfile.foundry 或模型白名单。目前 Foundry 的公开文档确认的是基础 beta；并未确认较新的 `mcp-client-2026-09-15` beta 功能可用。因而本客户端会在 Foundry 拒绝依赖新 beta 的固定工具列表、mcp_tool_listing 和内联 MCP 变更，即使推理由 Anthropic hosting。Foundry 请求应保持列表未固定，也不要回放或添加这些较新的原生 block。下面的示例纯本地编码一个基础 MCP 连接：

~~~rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicMcpConfig};
use lingxi_llm_client::protocol::{ChatRequest, FoundryDeployment, FoundryHosting, ProviderProfile};
use lingxi_llm_client::{
    CodecContext, EncodeRequest, FoundryClaudeCodec, RequestMode, WireCodec,
};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "my-foundry-resource",
        "profile_name": "foundry-claude",
        "protocol": "foundry_claude",
        "base_url": "https://my-resource.services.ai.azure.com/anthropic",
        "auth": "none",
        "models": [{
            "display_model": "Claude Opus",
            "request_model": "prod-opus-55",
            "billing_model": "accounting-opus"
        }]
    }))?;
    // 仅展示身份字段的写法；基础 MCP 配置不要求该字段。
    profile.models[0].foundry = Some(FoundryDeployment {
        hosting: FoundryHosting::Azure,
        model_id: "claude-opus-5-5".into(),
    });

    let mut request: ChatRequest = serde_json::from_value(json!({
        "model": "prod-opus-55",
        "messages": [{"role":"user","content":[{"type":"text","text":"Search the docs server"}]}]
    }))?;
    request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
        AnthropicMcpConfig::new("docs", "https://mcp.example.com/sse")?,
    ).into());

    let context = CodecContext::new(&profile, "prod-opus-55", RequestMode::Complete);
    FoundryClaudeCodec.validate_request(&request, &context)?;
    let wire = FoundryClaudeCodec.encode_request(EncodeRequest::new(&request), &context)?;
    let body: Value = serde_json::from_slice(&wire.body)?;
    assert_eq!(body["model"], "prod-opus-55");
    let beta = wire.headers.iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| value.as_str())
        .unwrap_or("");
    assert!(beta.split(',').any(|value| value == "mcp-client-2025-11-20"));
    Ok(())
}
~~~

Foundry 的模型目录、功能可用性、认证方式和部署区域可能变化。调用方负责提供凭证，以及获取和刷新 Microsoft Entra token；此 codec 只编码请求。以上示例属于本地编码检查，并未连接 Azure 资源或 Anthropic 账户。

参考：[Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)、[Microsoft Foundry 部署与认证指南](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-claude)、[Anthropic Tool Search 兼容模型](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)、[Anthropic Web Fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool) 和 [Anthropic MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)。

## Code Execution 与程序化工具

`AnthropicHostedTool::CodeExecution` 使用选中模型行的显式托管方式与底层模型身份。仅 Anthropic 托管且满足兼容表的模型允许执行，Azure 托管在发送前拒绝；wire `model` 仍使用部署名。程序化 caller 还需满足独立的模型限制。

非流式和流式响应保留原生容器与用量 metadata。通过 `AnthropicContainerScope::new_foundry` 导入返回的容器 ID，绑定 profile、精确资源端点、稳定账户身份、部署名与 `FoundryDeployment`；这些信息改变后不能继续复用。不确定的执行结果不会触发自动重试或切换。文件和 Skill 的支持边界见 [Code Execution](anthropic-code-execution.md)。
