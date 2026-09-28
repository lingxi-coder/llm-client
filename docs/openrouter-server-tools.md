# OpenRouter 服务端工具

客户端将 OpenRouter 的正则工具搜索和托管 Shell 暴露为带类型的
`HostedTool` 变体。这些工具由 OpenRouter 在请求期间执行，不会作为待宿主运行的
`ToolUse` 调用返回。模型可以在最终回答前调用它们零次或多次。该功能可通过
OpenRouter Responses 和 Messages API 使用。官方文档说明，工具搜索适用于任何模型和
提供商。Shell 同样适用于任意模型，但只支持全局 `openrouter.ai` 端点；区域端点和
Chat Completions 会拒绝此工具。详见[服务端工具概览](https://openrouter.ai/docs/guides/features/server-tools)、
[工具搜索契约](https://openrouter.ai/docs/guides/features/server-tools/tool-search)
和 [Shell 契约](https://openrouter.ai/docs/guides/features/server-tools/shell)。

请配置 `provider_id = "openrouter"`、协议为 `OpenAiResponses` 或
`AnthropicMessages`、base URL 为 `https://openrouter.ai/api/v1` 的 profile。模型必须能通过
此 profile 的目录解析。若 profile 需要凭证，请通过 `RequestOptions` 传入；以下示例接收
调用方准备好的 options，不会自行查找凭证。

## 工具搜索

添加 `OpenRouterHostedTool::ToolSearch`，并将希望模型搜索后再看到的函数工具的
`ToolSpec.defer_loading` 设为 `true`。发现后，函数仍由调用方执行：最终产生的
`ToolUse` 仍属于宿主。`max_results` 默认使用 OpenRouter 的默认值，上限为 50。本客户端
提供文档中的正则搜索变体；不提供 OpenRouter 的 Anthropic 别名或尚不支持的 BM25 变体。

延迟加载的工具必须配合 OpenRouter 工具搜索，或使用现有的 Anthropic 原生延迟加载路径。
使用 OpenRouter 管理延迟加载时，请使用自动工具选择。API 也接受 `allowed_tools` 选择，
但公开契约没有给出完整结构；本客户端没有对此建模，因此当 OpenRouter 搜索与延迟加载工具
同时使用时，会拒绝非 `Auto` 选择。Anthropic 原生搜索请求仍遵循原有的 Anthropic 工具选择行为。

```rust,no_run
use lingxi_llm_client::providers::openrouter::types::{OpenRouterToolSearchConfig};
use lingxi_llm_client::protocol::{ChatRequest, ConversationMessage, ToolChoice, ToolSpec};
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::LlmError;
use serde_json::{json, Value};

pub async fn discover_a_client_tool(
    client: &LlmClient,
    options: &RequestOptions,
) -> Result<(), LlmError> {
    let request = ChatRequest {
        controls: Default::default(),
        prompt_cache: Default::default(),
        output_format: Default::default(),
        model: "openai/gpt-5.2".into(),
        native_options: Vec::new(),
        hosted_tools: vec![lingxi_llm_client::providers::openrouter::native::OpenRouterHostedTool::ToolSearch(
            OpenRouterToolSearchConfig {
                max_results: Some(10),
            },
        ).into()],
        continuation: None,
        system: vec![],
        messages: vec![ConversationMessage::user_text("Find the weather tool and check Tokyo.")],
        tools: vec![ToolSpec {
    tool_type: None,
    extra: serde_json::Value::Null,
            name: "get_weather".into(),
            description: "Get the current weather for a city.".into(),
            input_schema: json!({
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
            }),
            strict: false,
            defer_loading: true,
            native_options: Vec::new(),
        }],
        tool_choice: ToolChoice::Auto,
        max_tokens: None,
        temperature: None,
        thinking: None,
        service_tier: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    };
    let _response = client
        .chat()
        .complete(&request, options)
        .await?;
    Ok(())
}
```

## 托管 Shell

`OpenRouterHostedTool::Shell` 会发送 `type: "openrouter:shell"`。省略参数时，使用
OpenRouter 默认值：`engine: "auto"`、临时 `container_auto` 环境、每条命令 120 秒超时、
每个输出流最多 16,384 个字符，并禁用网络。显式的 `timeout_ms` 不得超过 300,000，
`max_output_length` 不得超过 65,536。命令列表由模型生成；OpenRouter 每次调用最多允许
100 条命令。服务端会钳制文档中的数值上限，并拒绝超过命令数上限的调用。

网络策略可以省略、设为禁用，或设为最多含 50 条小写主机名/通配符模式的允许列表。条目
不能包含 scheme、路径或端口。省略策略时会禁用出站访问。容器启动后，其网络策略固定；
复用该容器时必须继续发送相同策略。

在 Messages API 中，Shell 调用和结果分别以原生的 `server_tool_use` 和
`openrouter_shell_tool_result` 内容块返回。在 Responses API 中，它们是提供商原生输出项。
编解码器会将这些内容保留为并回放为 `ProviderContent`，不会把服务端 Shell 命令作为宿主
需要执行的 `ToolUse` 返回。流式调用方应保留 `ProviderContent` 块，也可以检查附带的原始
`ProviderEvent` 帧。Messages 流中返回的容器信息位于原始 `message_start` 事件里；流的终止
事件不会重建 `ChatResponse::openrouter_container()` 字段。

### 复用容器

OpenRouter 按账号和 workspace 对持久容器进行隔离。本客户端要求调用方导入的
`OpenRouterContainerRef` 带有本地 `OpenRouterContainerScope`，其中绑定精确的 profile 名称、
端点和稳定的 `RequestOptions.account_scope`。复用引用的每次请求都要设置相同的
`account_scope`。该 scope 只保存在本地；线上仅发送提供商返回的容器 ID。

Messages 的 `ChatResponse::openrouter_container()` 原样保留提供商返回的 envelope，不会自动
将其信任或绑定到账号。调用方可以明确地将刚刚使用的 profile、端点和账号与观察到的 ID
关联：

```rust,no_run
use lingxi_llm_client::providers::openrouter::types::{OpenRouterContainerScope};
use lingxi_llm_client::protocol::{ChatRequest, };
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::LlmError;

pub async fn run_in_a_reused_container(
    client: &LlmClient,
    response: &lingxi_llm_client::protocol::ChatResponse,
    account_scope: &str,
    options: &RequestOptions,
) -> Result<(), LlmError> {
    let scope = OpenRouterContainerScope::new(
        "openrouter-global", // 替换为实际所选 profile 的精确名称
        "https://openrouter.ai/api/v1",
        account_scope,
    )?;
    let Some(container) = response
        .openrouter_container()
        .map(|metadata| metadata.reference_for(scope))
        .transpose()?
    else {
        return Ok(());
    };
    let request: ChatRequest = serde_json::from_value(serde_json::json!({
        "model":"openai/gpt-5.2",
        "messages":[{"role":"user","content":[{"type":"text","text":"Continue."}]}],
        "hosted_tools":[{
            "type":"openrouter_shell",
            "config":{"environment":{"type":"container_reference","container":container}}
        }]
    }))
    .map_err(|error| LlmError::InvalidRequest {
        message: error.to_string(),
    })?;
    let scoped_options = RequestOptions {
        account_scope: Some(account_scope.to_owned()),
        ..options.clone()
    };
    let _response = client.chat().complete(&request, &scoped_options).await?;
    Ok(())
}
```

`OpenRouterContainerMetadata::reference_for` 只导入字符串 ID；返回的 envelope 是否属于给定
scope，由宿主负责判断。本客户端不会在额外接口中创建容器或轮询容器，也不会重试可能已执行
服务端工具的请求。此类请求会固定到一次路由尝试，因为响应中断后重发可能重复提供商侧操作。

`openrouter:bash` 是另一个仅限 Messages 的服务端工具，本 API 不包含它。凭证仍由所选
OpenRouter profile 和 `RequestOptions` 提供；这些服务端工具不引入额外凭证字段。
