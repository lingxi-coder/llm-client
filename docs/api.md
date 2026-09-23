# API 接口文档

本文对应本仓库的 Rust API。`lingxi-llm-client` 是库，内置 HTTP 客户端，不提供监听端口的 HTTP 服务、CLI 或密钥管理服务。核心接口从 `lingxi_llm_client` 导入；请求、响应和配置类型从 `lingxi_agent_api::protocol` 导入。

## 目录

- [接入与生命周期](#接入与生命周期)
- [客户端和构建器](#客户端和构建器)
- [请求与消息](#请求与消息)
- [Web Search 接口](#web-search-接口)
- [流式响应](#流式响应)
- [Provider 配置与路由](#provider-配置与路由)
- [认证与凭证](#认证与凭证)
- [传输接口](#传输接口)
- [模型目录](#模型目录)
- [用量与费用](#用量与费用)
- [错误处理](#错误处理)
- [扩展接口](#扩展接口)

## 接入与生命周期

1. 在 Tokio 异步运行时中使用客户端。
2. 加载自己的 `ProviderProfile`，或使用 `builtin_providers()` / `merge_providers(user)`。
3. 用 `LlmClientBuilder::new(&profiles)?` 创建构建器，再调用 `build()`；已自动注册 API key 和 Bearer 认证器。
4. 宿主取得有效凭证，通过每次请求的 `RequestOptions` 传入。
5. 调用 `complete()` 或 `stream()`；宿主负责会话历史、工具执行、取消和后续请求。

依赖配置见 [README](../README.md#在其他-rust-项目中使用)。下面示例使用 `serde_json` 构造配置，调用项目还需要声明 `serde_json = "1"` 和 Tokio 运行时依赖。无需直接依赖 `reqwest`，也无需自行实现传输和时钟。

```rust,no_run
use lingxi_agent_api::protocol::{
    CompletionRequest, ConversationMessage, ProviderProfile, Secret, ToolChoice,
};
use lingxi_llm_client::{LlmClientBuilder, RequestOptions};

async fn ask(api_key: String) -> Result<String, Box<dyn std::error::Error>> {
    let profile: ProviderProfile = serde_json::from_value(serde_json::json!({
        "provider_id": "my-provider",
        "profile_name": "primary",
        "base_url": "https://api.example.com/v1",
        "protocol": "open_ai_chat",
        "auth": "api_key",
        "models": [{
            "display_model": "My Model",
            "request_model": "my-model",
            "billing_model": "my-model",
            "capabilities": {
                "vision": false, "documents": false, "tools": true,
                "reasoning": false, "signed_reasoning": false,
                "streaming": true, "structured_output": false
            }
        }]
    }))?;
    let client = LlmClientBuilder::new(&[profile])?.build()?;
    let request = CompletionRequest {
        model: "primary/my-model".into(),
        web_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![ConversationMessage::user_text("你好")],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: Some(1024),
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: serde_json::Value::Null,
    };
    let options = RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..RequestOptions::default()
    };
    let response = client.complete(&request, &options).await?;
    Ok(response.message.text())
}
```

`api.example.com` 和 `my-model` 是占位值，需替换为服务实际地址和模型 ID。函数不负责创建异步运行时；在 Tokio 运行时内调用，例如通过 `#[tokio::main]`。Tokio 可启用 `rt-multi-thread` 和 `macros` 特性。需要自定义传输或测试时，可使用 `LlmClientBuilder::with_transport(http, &profiles)`，并按需调用 `with_clock(clock)` 设置测试时钟。常规调用无需组装服务或处理时钟。

## 客户端和构建器

### `LlmClientBuilder`

| 方法 | 返回值 | 行为 |
| --- | --- | --- |
| `new(&[ProviderProfile])` | `Result<Self, LlmError>` | 创建内置 HTTP 传输和系统时钟，注册全部内置 codec、目录解析器，以及 `ApiKey` / `Bearer` 认证器 |
| `with_transport(Arc<dyn Transport>, &[ProviderProfile])` | `Self` | 使用自定义传输与内置系统时钟，注册全部内置 codec、目录解析器及 `ApiKey` / `Bearer` 认证器 |
| `with_clock(Arc<dyn Clock>)` | `&mut Self` | 覆盖构建器的时钟，适用于固定时间的测试 |
| `register_codec(Arc<dyn WireCodec>)` | `&mut Self` | 按协议族添加或替换 codec |
| `register_directory(Arc<dyn ModelDirectory>)` | `&mut Self` | 按目录协议形状添加或替换解析器 |
| `register_authenticator(AuthStrategy, Arc<dyn Authenticator>)` | `&mut Self` | 为指定认证策略添加或替换实现 |
| `add_profile(ProviderProfile)` | `&mut Self` | 追加连接配置 |
| `codec_families()` | `Vec<ProtocolFamily>` | 查询已注册的协议族 |
| `build(self)` | `Result<LlmClient, BuildError>` | 消费构建器并验证配置 |

`BuildError` 包括 `DuplicateProfile { profile_name }`、`MissingCodec { profile_name, family }`、`MissingAuthenticator { profile_name, strategy }` 和 `InvalidPeakSchedule { profile_name, reason }`。`AuthStrategy::None` 不需要认证器；缺少目录解析器不阻止构建。构建时会拒绝无效或空的峰值价格时间窗；构建成功不代表凭证有效、地址可达或 provider 支持所有请求参数。

### `LlmClient`

| 方法 | 返回值 | 行为 |
| --- | --- | --- |
| `complete(&CompletionRequest, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | 解析路由、编码、认证、发送并解码完整响应 |
| `stream(&CompletionRequest, &RequestOptions).await` | `Result<ModelStream, LlmError>` | 打开 HTTP 流，返回统一事件接口 |
| `web_search(&CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | 本次请求启用 provider 托管搜索，返回答案和来源 |
| `web_search_stream(&CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<ModelStream, LlmError>` | 本次请求启用搜索并返回流式事件 |
| `resolve(&str)` | `Result<ResolvedRoute, ResolveError>` | 在配置中解析模型及故障转移链 |
| `resolve_in(&str, Option<&str>)` | 同上 | 显式限制起始连接或连接组 |
| `models()` | `Vec<ModelListing>` | 返回静态配置模型，跳过 `connection.hidden` 连接 |
| `providers()` | `Vec<ProviderListing>` | 返回所有连接，包括隐藏连接和没有凭证的连接 |
| `profiles()` | `&[ProviderProfile]` | 读取客户端使用的配置 |
| `codec_families()` | `Vec<ProtocolFamily>` | 查询 codec 协议族 |
| `directory_shapes()` | `Vec<ProtocolFamily>` | 查询目录解析器支持的协议形状 |
| `directory_for(&ProviderProfile)` | `Option<Arc<dyn ModelDirectory>>` | 根据 `model_list` 查找目录解析器 |
| `estimate_cost(&ResolvedRoute, &Usage, Submission)` | `Result<Option<CostEstimate>, LlmError>` | 按配置价格和当前时钟估算费用 |
| `estimate_actual_cost(&ResolvedRoute, &CompletionResponse, &Usage, Submission)` | 同上 | 根据完整响应的实际成功连接估价 |
| `estimate_cost_for_profile(&ResolvedRoute, &str, &Usage, Submission)` | 同上 | 根据指定的成功连接估价，适用于流式响应 |

配置在构建时复制。更新模型或连接配置后，应重新构建客户端；读取目录本身不会修改 `models()` 的结果。

## 请求与消息

### `RequestOptions`

| 字段 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `credential` | `Option<Secret<String>>` / `None` | 本次请求的有效凭证；不读取配置中的环境变量或静态密钥 |
| `fallback_credentials` | `BTreeMap<String, Secret<String>>` / 空 | 按备用 profile 名提供独立凭证；未提供时不会复用首连接密钥 |
| `total_timeout` | `Option<Duration>` / `None` | 逐请求总时限；`complete()` 省略时默认 120 秒，`stream()` 省略时不设总时限 |
| `stream` | `bool` / `false` | 直接调用 codec 时控制编码模式；高层 `complete()` 强制非流式，`stream()` 强制流式 |

### `CompletionRequest`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `model` | `String` | 配置中的显示名、wire ID、alias 或限定模型引用 |
| `web_search` | `Option<WebSearchConfig>` | `None` 为默认关闭；`Some` 启用所选连接的托管搜索 |
| `previous_response_id` | `Option<ResponseId>` | Responses 续接的上一个响应 ID |
| `system` | `Vec<SystemBlock>` | 系统提示块：`text` 和 `cacheable` |
| `messages` | `Vec<ConversationMessage>` | 本轮输入及调用方维护的历史 |
| `tools` | `Vec<ToolSpec>` | 工具名称、说明、JSON Schema 与 `strict` 标志 |
| `tool_choice` | `ToolChoice` | `Auto`、`Any`、`None` 或 `Tool { name }` |
| `max_tokens` | `Option<u32>` | 输出 token 限制，由协议映射 |
| `temperature` | `Option<f32>` | 采样温度，provider 的支持范围由调用方确认 |
| `thinking` | `Option<ThinkingConfig>` | 思考配置，目前包含 `budget_tokens: Option<u32>` |
| `stop_sequences` | `Vec<String>` | 停止序列 |
| `metadata` | `serde_json::Value` | codec 按其实现处理的附加数据，不是通用透传任意参数的承诺 |

`CompletionRequest` 没有 `Default` 实现。通过 serde 读取时，`model`、`messages` 必填，其余有默认值或可省略。不同协议对工具选择、思考、多模态的表达不同，统一类型不保证每种组合都被所有服务接受。

`previous_response_id` 仅适用于 `OpenAiResponses` 且配置 `extra.supports_previous_response_id = true` 的端点。调用方保存响应 ID，并只发送需要新增的输入；客户端不自动保存会话。续接请求不会遍历故障转移连接，因为响应 ID 是端点侧状态。

### 消息和内容块

`ConversationMessage { role, content }` 的角色为 `User`、`Assistant`、`System`。便捷方法有 `user_text(text)`、`assistant(blocks)`、`text()` 和 `tool_uses()`；`text()` 只拼接文本块，不包含思考内容。

| `ContentBlock` 变体 | 字段 / 用途 |
| --- | --- |
| `Text` | `text`、可选 `thought_signature`；Gemini 重放需保留该字段 |
| `ToolUse` | `id: ToolUseId`、`name`、`input: Value`、可选 `provider_id` 与 `thought_signature`；Gemini 真实调用 ID 和签名需原样重放 |
| `ToolResult` | `tool_use_id`、`content`、`is_error`、可选 `blocks: Vec<Value>` |
| `Thinking` | `text`、可选 `signature`；重放时保留签名 |
| `RedactedThinking` | `data`；保留 provider 返回的原始值 |
| `Image` | `source: ImageSource`，支持 Base64 或 URL |
| `Document` | `source: DocumentSource` 与可选 `title`，支持 Base64、文本或 URL |
| `ProviderContent` | `protocol`、`value`；保留 Claude/DeepSeek 搜索的原生内容，以便下一轮重放 |

Base64 源携带 `media_type` 和 `data`；URL 源携带 `url`。库不执行工具、不自动下载附件，也不运行 `vision_delegate`。工具调用返回后，宿主执行工具，把带相同 `tool_use_id` 的结果加入下一轮输入。

### `CompletionResponse`

返回 `message: ConversationMessage`、`web_search: Option<WebSearchResult>`、`stop_reason: StopReason`、`usage: Usage`、`model: String`、`response_id: Option<ResponseId>` 和 `executed_profile: Option<String>`。高层 `complete()` 会设置实际成功的连接名称；直接调用 codec 解码时此字段为 `None`。`StopReason` 包括 `EndTurn`、`ToolUse`、`MaxTokens`、`StopSequence`、`Refusal` 和 `Other(String)`。

## Web Search 接口

`web_search()` 和 `web_search_stream()` 接受普通 `CompletionRequest`、本次搜索的 `WebSearchConfig` 和 `RequestOptions`。两种方法只克隆请求并设置 `web_search`，随后使用与 `complete()` / `stream()` 相同的路由、认证和故障转移逻辑；原请求不会被修改。传入的配置覆盖请求中已有的 `web_search`。也可以直接设置 `request.web_search = Some(config)` 后调用普通方法，效果相同。这里的“接口”是 Rust 客户端方法，本库不提供独立 HTTP 搜索服务。

```rust,no_run
use lingxi_agent_api::protocol::{CompletionRequest, ConversationMessage, LlmError, Secret, WebSearchConfig};
use lingxi_llm_client::{builtin_providers, LlmClientBuilder, RequestOptions};

async fn search(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let profiles = builtin_providers()?;
    let client = LlmClientBuilder::new(&profiles)?.build()?;
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "glm/glm-4.7",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "查找 Rust 最新版本，给出来源"}]}]
    }))?;
    let options = RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..RequestOptions::default()
    };
    let config = WebSearchConfig::default();
    let response = client.web_search(&request, config, &options).await?;
    println!("{}", response.message.text());
    if let Some(search) = response.web_search {
        for source in search.citations {
            println!("{}: {}", source.title.unwrap_or_default(), source.url);
        }
    }
    Ok(())
}
```

示例中的 `api_key` 由宿主提供；`glm` 连接使用 `ZHIPU_API_KEY` 对应的开放平台凭证。构建器不读取环境变量。其他预设可使用 `openai`、`anthropic`、`gemini`、`openrouter`、`zai`、`deepseek-search`、`kimi-search` 等支持搜索的连接，具体模型和账户权限由 provider 决定。

`WebSearchConfig` 各字段及返回值如下；省略所有字段表示让 provider 自行决定是否搜索和搜索范围。

| 字段 | 类型 / 默认值 | 作用 |
| --- | --- | --- |
| `allowed_domains` | `Vec<String>` / 空 | 只允许指定域名；不含 `https://` 或路径 |
| `blocked_domains` | `Vec<String>` / 空 | 排除指定域名；不能同时设置允许列表 |
| `max_uses` | `Option<u32>` / `None` | 单次请求最多搜索次数，需为正数；仅 Anthropic 适配器支持 |

`WebSearchResult.citations` 是 URL 与可选标题列表，`metadata` 保留 provider 原始搜索记录、引用位置、错误和用量。`web_search: None` 表示响应没有提供可识别的搜索元数据，不等同于客户端确认“没有发生搜索”。服务端决定是否调用搜索工具，启用配置并不保证本轮真的搜索。

流式调用：

```rust,no_run
use lingxi_agent_api::protocol::{CompletionRequest, LlmError, StreamEvent, WebSearchConfig};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn search_stream(client: &LlmClient, request: &CompletionRequest, options: &RequestOptions) -> Result<(), LlmError> {
    let mut stream = client.web_search_stream(request, WebSearchConfig::default(), options).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => print!("{text}"),
            StreamEvent::WebSearch { result } => {
                for source in result.citations {
                    eprintln!("source: {}", source.url);
                }
                // 如需准确引用位置或诊断搜索错误，保留 result.metadata。
            }
            StreamEvent::End { stop_reason, .. } => eprintln!("{stop_reason:?}"),
            _ => {}
        }
    }
    Ok(())
}
```

搜索事件可能分多帧到达；同一个 URL 可能在来源和正文引用中重复出现。`WebSearch` 是服务端元数据事件，无需执行客户端工具。使用 `ToolChoice::Auto` 对各搜索连接最通用；具体适配器对过滤、工具选择和模型的限制见[完整 Web Search 文档](web-search.md#支持矩阵与参数)。连接未声明 `extra.web_search`、协议不匹配或使用了该适配器不支持的参数时，会在 HTTP 请求前返回 `UnsupportedCapability`；非法域名和非正数搜索次数返回 `InvalidRequest`。搜索工具可能单独计费，`estimate_cost()` 只估算 token。

## 流式响应

```rust,no_run
use lingxi_agent_api::protocol::{CompletionRequest, LlmError, StreamEvent};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn read_stream(
    client: &LlmClient,
    request: &CompletionRequest,
    options: &RequestOptions,
) -> Result<(), LlmError> {
    let mut stream = client.stream(request, options).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => print!("{text}"),
            StreamEvent::End { stop_reason, .. } => eprintln!("{stop_reason:?}"),
            _ => {}
        }
    }
    if stream.usage_is_complete() {
        if let Some(usage) = stream.observed_usage() {
            eprintln!("tokens: {}", usage.total());
        }
    }
    Ok(())
}
```

`ModelStream` 提供自己的异步 `next()`，无需导入 `StreamExt`。可通过 `status()`、`header(name)`、`headers()` 读取打开流时的状态和响应头，header 查找不区分大小写。`executed_profile()` 返回接受请求的连接名称，可用于流结束后的费用估算。

| 事件 | 含义 |
| --- | --- |
| `Start { model, response_id }` | 响应开始；响应 ID 可能不存在 |
| `TextDelta { block, text }` | 文本增量 |
| `ReasoningDelta { block, text }` | 思考增量 |
| `ThoughtSignature { block, signature }` | 思考签名，需与对应 block 一起保留 |
| `RedactedThinking { block, data }` | 不透明思考数据 |
| `ToolCallDelta { block, id, name, arguments_fragment }` | 工具参数片段；按 block 累积后再解析 JSON |
| `WebSearch { result }` | 托管搜索来源及 provider 原生元数据；可能出现多次 |
| `End { stop_reason, usage }` | 解码器输出的结束事件 |

`observed_usage()` 可返回部分计数；`usage_is_complete()` 才表明计数完整且自洽，不能仅凭存在 `End` 就认定可用于计费。如果底层提前 EOF、尚未观察到 provider 的终止标记，解码器返回 `StreamInterrupted`。读取错误后应结束本轮。`ModelStream` 返回给调用方之后的内容错误不会自动切换连接，避免重复生成或重复执行工具。

丢弃 `ModelStream` 会丢弃底层字节流；宿主的 `Transport` 应使对应请求资源随之释放。SSE 响应应保留正确的 `Content-Type: text/event-stream`，库会重组任意网络 chunk；Bedrock decoder 自行重组 AWS event-stream 二进制帧。

## Provider 配置与路由

### `ProviderProfile`

必填字段是 `provider_id`、`profile_name`、`base_url`、`protocol`、`auth`。`provider_id` 是开放字符串，不需要为新厂商添加枚举。`profile_name` 是唯一连接名称。

| 字段 | 含义 / 默认行为 |
| --- | --- |
| `models` | `Vec<ModelProfile>`，默认空；模型需要先在配置中存在才能 resolve |
| `credential` | 凭证来源描述，默认 `None`，实际读取由宿主负责 |
| `model_list` | 省略表示与 `protocol` 相同；`"none"` 表示未发布；也可填另一个协议族字符串 |
| `connection` | 分组、顺序、隐藏和故障转移策略 |
| `pricing` | `billingMode`、`require_priced`、可选 `peak` |
| `signing` | 可选 `region`、`service`、`project`，供宿主签名实现使用 |
| `azure` | 可选 `api_version` 和 `deployment` |
| `info` | 展示名、说明、控制台/API key/文档链接和凭证提示 |
| `extra` | 协议选项、额外 body/header 等 JSON 数据 |
| `supports_websockets` 等 WebSocket 字段 | 配置元数据；高层客户端当前仍使用 HTTP |
| `visionDelegate` | serde 字段名；Rust 字段是 `vision_delegate`，当前客户端不自动委派 |

`ModelProfile` 必须指定 `display_model`、`request_model`、`billing_model`；可添加 `aliases`、`description`、`metadata`、`capabilities`、`pricing` 和模型级 `billing_mode`。三种模型名分别用于显示/匹配、请求协议和价格归属。`metadata` 保存上下文窗口、输出上限、模态等目录数据。

`ModelProfile.hidden` 默认为 `false`。设置为 `true` 后，`models()` 不再列出该模型，但它仍可通过模型名或 `resolve_in()` 显式调用。运行时可用 `set_model_visibility(profile_name, request_model, visible)` 修改并保存该设置。

### 协议与 URL

下表说明本库编码器实际追加的路径，不代表服务的在线可用性。`base_url` 不应已包含表中“追加路径”。

| `protocol`（serde 值） | codec | `base_url` / 追加路径 |
| --- | --- | --- |
| `open_ai_chat` | `OpenAiChatCodec` | 版本化根路径，如 `https://host/v1`；追加 `/chat/completions` |
| `open_ai_responses` | `OpenAiResponsesCodec` | 版本化根路径；追加 `/responses` |
| `anthropic_messages` | `AnthropicMessagesCodec` | 服务根路径；追加 `/v1/messages` |
| `gemini_generate_content` | `GeminiCodec` | 版本化根路径；追加 `/models/{model}:generateContent` 或流式 action |
| `azure_open_ai` | `AzureOpenAiCodec` | 资源根路径；追加 `/openai/deployments/{deployment}/chat/completions?api-version=...` |
| `foundry_claude` | `FoundryClaudeCodec` | 与 Anthropic 编码路径相同，追加 `/v1/messages` |
| `vertex_claude` | `VertexClaudeCodec` | 含项目和 location 的根路径；追加 `/publishers/anthropic/models/{model}:rawPredict` 或 `streamRawPredict` |
| `vertex_gemini` | `VertexGeminiCodec` | 含项目和 location 的根路径；追加 `/publishers/google/models/{model}:generateContent` 或流式 action |
| `bedrock_claude` | `BedrockClaudeCodec` | Bedrock runtime 根路径；追加 `/model/{model}/invoke` 或 `invoke-with-response-stream` |

Azure 必须配置 `azure.api_version`；`azure.deployment` 省略时使用 `request_model`。托管平台 codec 负责 URL 和 wire body，不自动获取云凭证或实现 AWS SigV4 签名。

### 解析和故障转移

`resolve()` 优先匹配完整 `display_model`、`request_model` 或 alias，因此合法的含 `/` 模型 ID 保持完整。完整匹配不存在时，再解析 `profile/model` 或 `group/model`。限定名同时是连接名和组名时，连接名优先。`resolve_in(model, Some(name))` 可显式选择起始连接或组。

跨组同名返回 `ResolveError::AmbiguousAcrossGroups`；单连接中多个模型匹配返回 `DuplicateOnProfile`；没有匹配返回 `UnknownModel`。请求方法会将解析错误转换为 `LlmError::ModelUnavailable`。

`ResolvedRoute` 包含 `provider_id`、`profile_name`、`request_model`、`display_model`、`pricing_model`、`capabilities`、`connection_chain` 和 `failover`。不要把任意手工构造的 route 当作客户端已经验证过的路由。

`connection.group` 省略时，组名为 `profile_name`。连接按 `(order, profile_name)` 排序。备选连接必须在同组、提供相同 `request_model` 且有效计费模式相同；限定起始连接不禁用该组的故障转移。未指定连接时优先选择可见连接；显式指定 profile 仍可选中隐藏连接，隐藏连接也可参与故障转移。

**默认不启用故障转移。** `FailoverTriggers::default()` / `NONE` 全为 false；显式赋值 `FailoverTriggers::DEFAULT` 可启用全部五类。JSON 示例：

```json
{
  "connection": {
    "group": "my-provider",
    "order": 0,
    "failover": {
      "rateLimit": true,
      "overloaded": true,
      "serverError": true,
      "network": true,
      "auth": false
    }
  }
}
```

该片段需合并进完整 `ProviderProfile`。`rateLimit` 匹配限流和额度耗尽，`network` 匹配传输错误和超时；上下文超限、不支持能力、TLS 错误不触发切换。策略取自起始 route，每个连接至多尝试一次；库不执行退避等待或同连接重试。

### 内置配置与扩展参数

`builtin_providers() -> Result<Vec<ProviderProfile>, PresetError>` 返回编译进库的静态目录。`merge_providers(user)` 保留用户条目，并追加未被同名用户条目覆盖的预设；这是整条 profile 覆盖，不是字段合并。用户输入中的重复名称仍由 builder 拒绝。`PresetError` 分为解析失败 `Invalid` 和空模型列表 `NoModels`。

### 本地保存与多账号

`LlmClient` 创建后可调用 `set_config_dir(path)`。库在该目录管理 `providers.json`，立即加载其中的 profile，并将目录固定为绝对路径，后续工作目录变化不会改变保存位置；同名本地配置覆盖 builder 的初始配置。再次设置目录时会清除上一个目录加载的配置。多个客户端写入同一目录时，库使用 `.providers.json.lock` 协调写入，并在锁内读取最新配置后应用本次修改；直接修改 JSON 的外部程序也需要遵守这个锁。`sync_provider(profile_name, credential).await` 使用该连接自己的凭证读取模型目录，按 `request_model` 更新现有模型、加入新模型，再保存配置。目录缺失、请求失败或写盘失败时，旧文件与内存配置保持不变。同步不会删除目录未返回的旧模型，也不会改变已有模型的 `hidden`、价格、能力或别名。

| 操作 | API | 保存行为 |
| --- | --- | --- |
| 新增 / 修改 | `add_provider(profile)` | 按 `profile_name` 新增或整条替换并写盘；也可用于多账号中的单个账号 |
| 查询 | `provider(profile_name)`、`profiles()`、`providers()`、`deleted_builtin_profiles()` | 读取配置、列表摘要及已软删除内置项的名称 |
| 删除 | `remove_provider(profile_name)` | 自定义 profile 从文件删除；内置 profile 记录软删除，重启后仍停用 |
| 恢复内置项 | `restore_builtin(profile_name)` | 清除软删除标记并保存库内预设；即使 builder 有同名自定义覆盖也以预设为准 |

删除的是单个连接账号，不会连带删除同组的其他账号；如需停用整个 provider，逐个删除其 profile。重加一个已软删除的同名配置也会清除软删除标记。自定义 profile 若由宿主在每次启动时再次传给 builder，宿主还须从自己的输入中移除它。

`set_tracked_models(provider_id, request_model_ids)` 为同一家 provider 的所有账号设置全局模型白名单，并保存在 `providers.json`。未配置白名单的 provider 不显示或保存任何模型；目录同步只合并白名单中的模型，保存时也只写入白名单中的模型。先调用 `add_provider` 再设置白名单也能使用本次会话提供的模型；重启后若要扩大白名单，需要由 builder 再次提供这些模型或重新同步目录。如果其他客户端修改了同名 profile 的模型列表，本客户端不会再用旧缓存补回被删除的模型。`tracked_models(provider_id)` 可读取当前名单；`untrack_model(provider_id, request_model)` 从名单删除单个模型，再次同步也不会重新加入。白名单决定是否跟踪、保存和显示；模型级 `hidden` 只决定已跟踪模型是否出现在 `models()` 中。

同一家 provider 的每个账号使用独立的 `profile_name` 和 `connection.connection_id`，并设置相同的 `connection.group`。主账号设置 `hidden: false`，备用账号设置 `hidden: true`，按 `order` 排序。`sync_provider` 逐账号执行，模型目录不在账号间复制。故障转移需要在 `RequestOptions.fallback_credentials` 中按备用 `profile_name` 提供其密钥；配置文件仅保存 `env` 或 `host_managed` 等凭证来源，不保存明文密钥。`CredentialConfig::Static` 会被保存接口拒绝。

```rust,no_run
use lingxi_agent_api::protocol::{ProviderProfile, Secret};
use lingxi_llm_client::LlmClientBuilder;

# async fn example(primary: ProviderProfile, spare: ProviderProfile) -> Result<(), Box<dyn std::error::Error>> {
let mut client = LlmClientBuilder::new(&[])?.build()?;
client.set_config_dir("./config")?;
client.set_tracked_models("acme", ["model-id".to_owned()])?;
client.add_provider(primary)?;
client.add_provider(spare)?;
client.sync_provider("primary", Some(&Secret::new("key".to_owned()))).await?;
client.set_model_visibility("primary", "model-id", false)?;
assert!(client.provider("primary").is_some());
client.untrack_model("acme", "model-id")?;
client.remove_provider("primary")?;
# Ok(())
# }
```

`ProviderStoreError` 区分文件、JSON、配置验证、目录请求和分页错误。同步最多读取 100 页，并拒绝重复 cursor；若请求期间连接配置被其他客户端修改，同步返回 `ProfileChanged` 且不写入旧连接的模型。启动后如需使用已保存的 provider，须再次调用 `set_config_dir`；本库不自动读取环境变量中的凭证。

`extra.body` 用于补充协议未写入的 JSON 字段，不能覆盖已写字段、模型身份或续接 ID；`extra.headers` 用于非凭证附加头，不能覆盖 codec 已写头。认证头应交由认证器。

常用协议选项包括 `credential_header`（API key 所在头）、`stream_usage_opt_in`、`preserve_reasoning_content`、`thinking_rejects_forced_tool_choice` 和 `supports_previous_response_id`。这些选项由对应 codec 读取，不保证对每个协议生效。

## 认证与凭证

内置认证器均为无状态 unit struct。`new()` 和 `with_transport()` 均自动为 `ApiKey`、`Bearer` 注册对应实现；也可通过 `register_authenticator(..., Arc::new(...))` 替换：

- `ApiKeyAuthenticator`：Anthropic 系列默认 `x-api-key`，Gemini 系列默认 `x-goog-api-key`，OpenAI 系列默认 `authorization: Bearer ...`。`extra.credential_header` 可覆盖头名，例如 Azure API key 使用 `api-key`。
- `BearerAuthenticator`：始终使用 `authorization: Bearer ...`，适用于调用方已经获得的有效 token。

`AuthStrategy` 包括 `ApiKey`、`Bearer`、`OAuthBearer`、`CopilotBearer`、`ChatGptOAuth`、`GcpToken`、`AzureToken` 和 `None`。策略名并不自动启用登录、刷新或交换逻辑；除构建器已注册的 `ApiKey` / `Bearer` 外，宿主需为使用的策略注册适合的实现。

`CredentialConfig::{Env, Static, HostManaged, None}` 只是来源描述；内置认证器仅使用 `RequestOptions.credential`，缺失时返回 `Authentication`。`Secret<String>` 的 Debug/Display 脱敏且不能序列化，读取明文需 `expose_secret()` 或 `into_inner()`；它不承诺内存清零。认证后的 `HttpRequest.headers` 包含明文凭证；`HttpRequest` 的 Debug 输出会隐藏 URL、header 值与 body 内容，但直接记录这些字段仍需宿主自行脱敏。

`RequestOptions.credential` 只用于首连接。自动故障转移到需要认证的备用连接时，必须在 `fallback_credentials` 中按 profile 名提供其凭证；缺失时返回 `Authentication`，不会发送该备用请求。宿主负责凭证的获取和更新；自定义认证器仍可按传入的 `profile` 实现其他认证策略。

## 传输接口

### 内置 HTTP 客户端

`HttpTransport::new() -> Result<Self, LlmError>` 创建可复用的 HTTP 客户端，基于 reqwest 0.12、Rustls TLS 与流式响应。调用方无需导入 reqwest；执行网络请求需要 Tokio 运行时。`LlmClientBuilder::new(&profiles)?` 自动创建 HTTP 客户端和 `SystemClock`，无需额外服务组装。单独调用模型目录接口时，可直接使用 `HttpTransport`。

| 行为 | 内置实现 |
| --- | --- |
| HTTPS | 使用 Rustls 校验证书 |
| 重定向 | 默认禁止跟随；`execute` 和 `execute_no_follow` 都保留原始重定向响应 |
| 重试 | 不自动重试；高层显式配置的故障转移仍可生效 |
| 连接超时 | 30 秒 |
| 流读取空闲超时 | 默认 60 秒；`HttpTransport::with_read_timeout(Duration)` 可在客户端级调整 |
| 总超时 | `complete()` 默认 120 秒；`stream()` 默认不设总时限，活跃长流可持续运行；`RequestOptions.total_timeout` 为两者指定总时限，包含流读取 |
| HTTP 错误状态 | 保留状态、响应头和 body，由 codec 分类，不提前丢弃错误体 |
| 网络错误 | 返回语义化 `LlmError`，错误消息不包含请求 URL、认证头或 body |
| WebSocket | 不支持；扩展方法返回不支持能力错误 |

复用客户端可复用连接池；丢弃响应流会释放对应资源。流式请求的总时限不会因收到新 chunk 而重置。需要不同网络策略时，可继续注入自定义 `Transport`。

### 自定义传输与时钟

`LlmClientBuilder::with_transport(http, &profiles)` 接收 `Arc<dyn Transport>`，同时使用内置 `SystemClock`。`Clock::now() -> SystemTime` 用于价格时段计算；测试可通过构建器的 `with_clock(Arc<dyn Clock>)` 覆盖系统时钟。普通使用无需实现或导入 `Clock`。

`Transport: Send + Sync + 'static` 使用 `async_trait`，需要实现：

| 方法 | 返回类型 | 责任 |
| --- | --- | --- |
| `execute(HttpRequest)` | `Result<HttpResponse, LlmError>` | 完整响应，包括非 2xx 的状态、头和 body，交给 codec 分类 |
| `execute_no_follow(HttpRequest)` | 同上 | 默认调用 `execute`；需要禁止重定向的宿主必须重写 |
| `open_stream(HttpRequest)` | `Result<StreamResponse, LlmError>` | 保留状态和头，body 是 `BoxStream<'static, Result<Bytes, LlmError>>` |
| `open_responses_websocket_session(HttpRequest)` | `Result<Box<dyn WebSocketSession>, LlmError>` | WebSocket 扩展入口；高层客户端当前不自动调用 |

`HttpRequest` 包含 `method`、`url`、`headers: Vec<(String, String)>`、`body: Bytes`、`timeout: Option<Duration>`。`HttpResponse` 包含 `status: u16`、`headers`、`body`，并提供不区分大小写的 `header()`。`StreamResponse` 也提供 `header()`。

传输实现应负责 TLS、代理、连接池、超时、网络错误归类、响应资源释放和重定向策略。流的非 2xx 错误 body 最多收集 64 KiB，再由 codec 分类。不得把错误 HTTP 状态预先丢弃为不含响应体的通用网络错误。

`WebSocketSession: Send` 有异步 `send(WsMessage)`、`recv() -> Result<Option<WsMessage>, LlmError>` 和 `close()`；`WsMessage` 为 `Text(String)` 或 `Binary(Bytes)`。`UrlOpener::open(&str) -> Result<(), String>` 是独立宿主能力，不由客户端构建器管理。

## 模型目录

内置 `OpenAiChatDirectory`、`AnthropicMessagesDirectory` 和 `GeminiDirectory`。不存在对应 reader 或配置 `model_list = "none"` 时，`directory_for()` 返回 `None`，并不表示现有模型不可调用。Responses 端点若使用 OpenAI models 列表，应显式配置 `model_list = "open_ai_chat"`。

目录操作是显式的：构造请求 → 认证 → 传输 → 解码 → 使用 cursor 获取下一页。以下示例适用于传入 API key 的 profile；其他认证策略应传入匹配的认证器。

```rust,no_run
use lingxi_agent_api::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{Authenticator, ApiKeyAuthenticator, LlmClient, LiveModel, Transport};

async fn list_live_models(
    client: &LlmClient,
    http: &dyn Transport,
    profile: &ProviderProfile,
    key: &Secret<String>,
) -> Result<Vec<LiveModel>, LlmError> {
    let directory = client.directory_for(profile).ok_or_else(|| LlmError::UnsupportedCapability {
        message: "该连接没有可用的模型目录".into(),
    })?;
    let mut models = Vec::new();
    let mut cursor = None;
    loop {
        let mut request = directory.list_request(profile, cursor.as_deref());
        ApiKeyAuthenticator.apply(&mut request, profile, Some(key)).await?;
        let response = http.execute(request).await?;
        let page = directory.decode_page(&response)?;
        models.extend(page.models);
        cursor = page.next_cursor;
        if cursor.is_none() {
            return Ok(models);
        }
    }
}
```

生产宿主还应设置总刷新时限/页数限制，避免异常服务反复返回相同 cursor。内置目录请求单页超时为 30 秒，由传输实现执行。

`ModelPage { models, next_cursor }` 的 cursor 是不透明字符串，原样传回。`LiveModel` 包含 `request_model`、可选 `display_name`、`description`、`context_window` 和 `max_output_tokens`，不包含价格。目录返回的模型不会自动加入配置；宿主负责与静态目录合并并重新构建客户端。

## 用量与费用

`Usage` 的 `input_tokens`（未缓存输入）、`output_tokens`、`cache_read_tokens` 和 `cache_write_tokens` 是互斥计费桶。`reasoning_tokens` 已包含在输出 token 内，不能再相加。`Usage::total()` 求四个计费桶之和。

`Usage.cost: Option<ReportedCost>` 是 provider 实际报告的金额。`ReportedCost.nano_usd` 以十亿分之一美元为单位；`from_usd(f64)` 拒绝负数、非有限数和溢出值。金额缺失表示未知，不是免费。

```rust,no_run
use lingxi_agent_api::protocol::{LlmError, Submission, Usage};
use lingxi_llm_client::LlmClient;

fn show_estimate(client: &LlmClient, model: &str, usage: &Usage) -> Result<(), LlmError> {
    let route = client.resolve(model)?;
    match client.estimate_cost(&route, usage, Submission::Interactive)? {
        Some(cost) => println!("estimated USD: {}", cost.total_usd),
        None => println!("目录没有该模型的价格"),
    }
    Ok(())
}
```

`CostEstimate` 从 `lingxi_llm_client::client::pricing` 导入，包括 `pricing_model`、`submission`、五个费用字段（input/output/cache_read/cache_write/reasoning）、`total_usd` 和价格 `source`。

- `TokenPricing` 每项费率以 USD / 百万 token 为单位；非零用量缺少对应费率时返回 `CostUnavailable`。使用中的费率必须非负且有限，费用计算溢出也返回 `CostUnavailable`。
- 整个模型缺少价格且 `pricing.require_priced = false` 时返回 `Ok(None)`，为 true 时返回错误。
- `Submission::Batch` 使用各桶显式的批处理费率，不假定固定折扣，也不表示客户端会提交 batch 作业。
- 如果单独配置 reasoning 价格，则从输出桶中拆出 reasoning，避免重复计费；`reasoning_tokens > output_tokens` 返回 `CostUnavailable`。
- `PeakSchedule` 使用 UTC 时间窗（`HH:MM-HH:MM`，结束时间可为 `24:00`）和可选工作日约束；高级调用方可使用 `pricing::estimate(...)` 显式传入时间和费率。

费用是目录估算，不替代 provider 账单。`estimate_cost` 只用于请求前对初始 route 预估；故障转移后应使用 `estimate_actual_cost(&route, &response, &response.usage, submission)`。流式响应可读取 `stream.executed_profile()`，汇总流事件中的 `Usage` 后调用 `estimate_cost_for_profile(...)`。两种实际连接估价仍使用目录价格，可能与 provider 账单不同。

## 错误处理

`LlmError` 来自共享 protocol，`kind()` 返回可用于分类表的 `LlmErrorKind`。不要依赖 `message` 文本匹配控制流程。

| 变体 | 含义 / 宿主处理方向 |
| --- | --- |
| `Authentication`、`OAuthRefreshDead`、`PermissionDenied` | 检查凭证、刷新或权限 |
| `InvalidRequest`、`UnsupportedCapability` | 调整请求或配置 |
| `RateLimited { retry_after, .. }`、`QuotaExceeded` | 限流/额度；退避或展示限制 |
| `ContextOverflow { limit, actual, .. }`、`RequestTooLarge` | 缩减历史或请求体 |
| `ModelUnavailable` | 模型无法解析或不可用 |
| `ProviderInternal`、`Overloaded` | provider 内部故障/过载 |
| `Transport`、`TransportTimeout`、`TlsCert` | 网络、超时、TLS 问题 |
| `StreamInterrupted` | 流内容损坏或意外中断 |
| `CostUnavailable` | 价格数据不足 |
| `MediaDelegationUnavailable`、`MediaDelegationPartial` | 共享错误类型，为宿主媒体委派提供语义 |

上述变体除特别列出的附加字段外均含 `message: String`。`retry_after` 是可选 `Duration`；当前内置解析器支持秒数形式的 `Retry-After`。`triggers_reactive_compaction()` 仅对 `ContextOverflow` 和 `RequestTooLarge` 返回 true，库本身不执行压缩或重试。

## 扩展接口

| 接口 | 必须实现的方法 | 注册 / 使用位置 |
| --- | --- | --- |
| `WireCodec` | `family`、`encode_request`、`decode_response`、`stream_decoder`、`response_usage` | builder 的 `register_codec`；同协议族后注册覆盖前者 |
| `StreamDecoder` | `decode_frame`、`finish`、`observed_usage`、`usage_is_complete`、`set_provider_metadata` | codec 每个响应创建独立有状态 decoder |
| `Authenticator` | 异步 `apply(&mut HttpRequest, &ProviderProfile, Option<&Secret<String>>)` | builder 按认证策略注册，编码完成后调用 |
| `ModelDirectory` | `shape`、`list_request`、`decode_page` | builder 按目录 shape 注册 |
| `FrameStream` | 异步 `next_frame() -> Result<Option<Bytes>, LlmError>` | 可供自定义集成使用，高层 `ModelStream` 使用字节流 |

`ProtocolFamily` 是封闭枚举；现有兼容协议的新 provider 只需配置，新增协议族需要修改枚举与 codec。codec 应把非成功响应转换为语义化 `LlmError`，归一化 token 用量，并保留思考签名和工具 ID。

`framing::sse::SseFrameSplitter` 提供 `new()`、`push(&[u8])` 和 `finish()`；未完成事件超过 8 MiB 时返回错误。`framing::eventstream` 提供 AWS 二进制帧解析并限制单帧大小。通常只在自定义传输/codec 集成时直接使用。

源码索引：[客户端](../src/client/mod.rs)、[共享请求/响应](../crates/agent-api/src/protocol/llm.rs)、[配置](../crates/agent-api/src/protocol/provider.rs)、[传输](../src/transport.rs)、[集成测试](../tests/)。可在本地生成逐项 Rust API 文档：

```sh
cargo doc --workspace --no-deps --open
```
