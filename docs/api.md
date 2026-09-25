# API 接口文档

[English](api.en.md)

本文对应本仓库的 Rust API。`lingxi-llm-client` 是库，内置 HTTP 客户端，不提供监听端口的 HTTP 服务、CLI 或密钥管理服务。核心客户端 API 可从 `lingxi_llm_client` 导入；请求、响应和配置类型通过 `lingxi_llm_client::protocol` 导入。

图像生成与编辑使用独立的 `client.images()` 服务；请求、任务和内置 provider 能力见[图像生成指南](images.md)。`client.chat()` 提供与原有 `complete()` / `stream()` 相同的对话接口。

## Region 区域过滤

构建 client 必须显式调用 `.with_region(Region::ChinaMainland)` 或 `.with_region(Region::International)`，否则 `build()` 返回 `BuildError::MissingRegion`。`Region` 从 `lingxi_llm_client::protocol` 导入；`client.region()` 返回当前选择。区域在 client 生命周期内固定，切换时重新构建 client，并可复用同一配置目录。

`ProviderProfile.regions` 声明使用区域，例如 TOML 中 `regions = ["china_mainland"]`；两区共享时填写 `["china_mainland", "international"]`。当前格式中 regions 缺省时两区可用，显式 `[]` 则两区均不可用。模型继承所属 profile 的区域，`ProviderListing` 和 `ModelListing` 都返回 `regions`。

`providers()` 和 `models()` 先按区域过滤，再应用各自现有的隐藏状态和模型白名单规则。模型解析、限定 profile/group 的调用、普通与流式请求、搜索及备用链都受区域限制；其他区域的同名模型不参与歧义判断。区域内的隐藏备用账号仍可用于故障切换。区域声明不保证网络可达性，不根据 IP、语言或 URL 自动推断，也不会改写 API 地址。

`provider()`、`profiles()` 是完整配置管理视图；CRUD、同步、账号 usage 和独立文件管理仍可操作其他区域账号。过滤不删除配置或模型，当前 client 的区域不写入共享 `providers.json`。当前格式未声明 `regions` 时两区可用；恢复内置配置会恢复其显式地区声明。

内置国内 profile：`qwen`、`qwen-search`、`minimax`、`kimi`、`kimi-search`、`glm`、`glm-coding`。`deepseek`、`deepseek-search`、`kimi-code` 两区共享；其他内置 profile 属于国际区域，包括 Qwen 香港、新加坡、美国及对应搜索连接。


## 目录

- [接入与生命周期](#接入与生命周期)
- [客户端和构建器](#客户端和构建器)
- [请求与消息](#请求与消息)
- [宿主工具执行与上下文恢复](#宿主工具执行与上下文恢复)
- [Web Search 接口](#web-search-接口)
- [Qwen 知识库 File Search](#qwen-知识库-file-search)
- [流式响应](#流式响应)
- [Provider 配置与路由](#provider-配置与路由)
- [认证与凭证](#认证与凭证)
- [传输接口](#传输接口)
- [模型目录](#模型目录)
- [用量与费用](#用量与费用)
- [账户额度与账户 Token 用量](#账户额度与账户-token-用量)
- [错误处理](#错误处理)
- [扩展接口](#扩展接口)

## 接入与生命周期

1. 在 Tokio 异步运行时中使用客户端。
2. 加载自己的 `ProviderProfile`，或使用 `builtin_providers()` / `merge_providers(user)`。
3. 用 `LlmClientBuilder::new(&profiles)?` 创建构建器，通过 `with_region(Region::International)` 或 `with_region(Region::ChinaMainland)` 选择区域，再调用 `build()`；已自动注册 API key 和 Bearer 认证器。
4. 宿主取得有效凭证，通过每次请求的 `RequestOptions` 传入。
5. 调用 `client.chat().complete()` 或 `client.chat().stream()`；宿主负责会话历史、工具执行、取消和后续请求。

依赖配置见 [README](../README.md#1-创建应用并添加依赖)。下面示例使用 `serde_json` 构造配置，调用项目还需要声明 `serde_json = "1"` 和 Tokio 运行时依赖。无需直接依赖 `reqwest`，也无需自行实现传输和时钟。

```rust,no_run
use lingxi_llm_client::protocol::{
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
            "capability_support": {
                "vision": "unknown", "documents": "unknown", "tools": "supported",
                "reasoning": "unknown", "signed_reasoning": "unknown",
                "streaming": "supported", "structured_output": "unknown"
            }
        }]
    }))?;
    let client = LlmClientBuilder::new(&[profile])?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
    let request = CompletionRequest {
        controls: Default::default(),
        service_tier: None,
        model: "my-model".into(),
        web_search: None,
        file_search: None,
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
    let response = client.chat().complete_in("primary", &request, &options).await?;
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
| `with_region(Region)` | `Self` | 消费并返回 builder，设置必选的使用区域 |
| `with_clock(Arc<dyn Clock>)` | `&mut Self` | 覆盖构建器的时钟，适用于固定时间的测试 |
| `register_codec(Arc<dyn WireCodec>)` | `&mut Self` | 按协议族添加或替换 codec |
| `register_directory(Arc<dyn ModelDirectory>)` | `&mut Self` | 按目录协议形状添加或替换解析器 |
| `register_authenticator(AuthStrategy, Arc<dyn Authenticator>)` | `&mut Self` | 为指定认证策略添加或替换实现 |
| `register_account_source(provider_id, AccountIdentity, Arc<dyn AccountUsageSource>)` | `&mut Self` | 按供应商及账户身份添加或替换只读账户数据源 |
| `register_profile_account_source(profile_name, AccountIdentity, Arc<dyn AccountUsageSource>)` | `&mut Self` | 将已登录账户源绑定到单个连接，优先于供应商级账户源 |
| `add_profile(ProviderProfile)` | `&mut Self` | 追加连接配置 |
| `codec_families()` | `Vec<ProtocolFamily>` | 查询已注册的协议族 |
| `build(self)` | `Result<LlmClient, BuildError>` | 消费构建器并验证配置 |

`BuildError` 包括 `MissingRegion`、 `DuplicateProfile { profile_name }`、`MissingCodec { profile_name, family }`、`MissingAuthenticator { profile_name, strategy }` 和 `InvalidPeakSchedule { profile_name, reason }`。`AuthStrategy::None` 不需要认证器；缺少目录解析器不阻止构建。构建时会拒绝无效或空的峰值价格时间窗；构建成功不代表凭证有效、地址可达或 provider 支持所有请求参数。

### `ChatService`

`client.chat()` 返回借用客户端的 `ChatService<'_>`，用于对话调用。它与仍然保留的顶层对话和流式方法共用配置、凭据、路由、附件准备、请求时限及故障转移逻辑，不保存会话历史或执行工具。

| 方法 | 返回值 | 行为 |
| --- | --- | --- |
| `models()` | `Vec<ModelListing>` | 列出当前区域可见模型，并过滤元数据中声明 `image` 输出的模型 |
| `complete(&CompletionRequest, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | 按模型路由并返回完整响应 |
| `complete_in(&str, &CompletionRequest, &RequestOptions).await` | 同上 | 限定起始 profile 或连接组 |
| `stream(&CompletionRequest, &RequestOptions).await` | `Result<ModelStream, LlmError>` | 按模型路由并打开流 |
| `stream_in(&str, &CompletionRequest, &RequestOptions).await` | 同上 | 在指定 profile 或连接组上打开流 |

托管搜索可在调用 Chat 前设置 `CompletionRequest.web_search` 或 `file_search`，也可使用 `LlmClient::web_search*` 便捷方法；`ChatService` 本身没有 `web_search*` 方法。配置、账户查询、路由检查、token 估算和价格接口仍属于 `LlmClient`；图像生成与编辑使用 `client.images()`。

### `LlmClient`

| 服务入口 | 返回值 | 用途 |
| --- | --- | --- |
| `chat()` | `ChatService<'_>` | 对话模型列表、完整响应与流式响应 |
| `images()` | `ImageService<'_>` | 独立图像目录、生成、编辑与原生任务 |

| 方法 | 返回值 | 行为 |
| --- | --- | --- |
| `complete(&CompletionRequest, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | 解析路由、编码、认证、发送并解码完整响应 |
| `complete_in(&str, &CompletionRequest, &RequestOptions).await` | 同上 | 显式指定起始 profile 或连接组；模型解析、凭证和执行使用同一路由 |
| `stream(&CompletionRequest, &RequestOptions).await` | `Result<ModelStream, LlmError>` | 打开 HTTP 流，返回统一事件接口 |
| `stream_in(&str, &CompletionRequest, &RequestOptions).await` | 同上 | 在指定 profile 或连接组上打开流 |
| `web_search(&CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | 本次请求启用 provider 托管搜索，返回答案和来源 |
| `web_search_stream(&CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<ModelStream, LlmError>` | 本次请求启用搜索并返回流式事件 |
| `web_search_in(&str, &CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | 在指定 profile 或连接组上执行搜索 |
| `web_search_stream_in(&str, &CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<ModelStream, LlmError>` | 在指定 profile 或连接组上流式搜索 |
| `resolve(&str)` | `Result<ResolvedRoute, ResolveError>` | 在配置中解析模型及故障转移链 |
| `resolve_in(&str, Option<&str>)` | 同上 | 显式限制起始连接或连接组 |
| `region()` | `Region` | 返回构建时选择的使用区域 |
| `models()` | `Vec<ModelListing>` | 返回当前区域有效配置中的模型（包括已同步的目录模型），跳过 `connection.hidden` 连接 |
| `providers()` | `Vec<ProviderListing>` | 返回当前区域的连接，包括隐藏连接和没有凭证的连接 |
| `account_usage(&str, &AccountQuery).await` | `Result<AccountSnapshot, AccountUsageError>` | 查询一个连接的账户额度与用量 |
| `accounts_usage(&BTreeMap<String, AccountQuery>).await` | `Vec<(String, Result<AccountSnapshot, AccountUsageError>)>` | 逐连接查询，保留独立结果 |
| `register_profile_account_source(&str, AccountIdentity, Arc<dyn AccountUsageSource>)` | `Result<(), ProviderStoreError>` | 配置变更后为连接重新绑定登录会话 |
| `profiles()` | `&[ProviderProfile]` | 读取客户端使用的配置 |
| `codec_families()` | `Vec<ProtocolFamily>` | 查询 codec 协议族 |
| `directory_shapes()` | `Vec<ProtocolFamily>` | 查询目录解析器支持的协议形状 |
| `directory_for(&ProviderProfile)` | `Option<Arc<dyn ModelDirectory>>` | 根据 `model_list` 查找目录解析器 |
| `estimate_local_tokens(&CompletionRequest)` | `Result<LocalTokenEstimate, LocalTokenCountError>` | 根据首选路由及精确模型 ID，离线估算可见的输入内容 |
| `estimate_local_tokens_in(&str, &CompletionRequest)` | 同上 | 限定 profile 或连接组后估算输入内容 |
| `price_quote(&str, Option<&str>, &PricingContext)` | `Result<PriceQuote, LlmError>` | 查询模型与档位的价格、来源和匹配条件 |
| `estimate_stream_cost(&ResolvedRoute, &ModelStream, Submission)` | `Result<CostEstimate, LlmError>` | 使用流的实际连接、档位和发送时间估价 |
| `estimate_cost(&ResolvedRoute, &Usage, &PricingContext)` | `Result<CostEstimate, LlmError>` | 按配置价格和当前时钟估算费用 |
| `estimate_actual_cost(&ResolvedRoute, &CompletionResponse, Submission)` | 同上 | 根据完整响应的实际成功连接估价 |
| `estimate_cost_for_profile(&ResolvedRoute, &str, &UsageReport, &InferenceReport, Submission)` | 同上 | 根据指定的成功连接和实际档位估价 |

配置在构建时复制。本地配置管理接口可以更新客户端的有效配置；直接调用目录解析器读取数据不会修改 `models()` 的结果。

## 请求与消息

### `RequestOptions`

| 字段 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `credential` | `Option<Secret<String>>` / `None` | 本次请求的有效凭证；不读取配置中的环境变量或静态密钥 |
| `fallback_credentials` | `BTreeMap<String, Secret<String>>` / 空 | 按备用 profile 名提供独立凭证；未提供时不会复用首连接密钥 |
| `total_timeout` | `Option<Duration>` / `None` | 逐请求总时限；`complete()` 省略时默认 120 秒（含视频的请求默认两小时），`stream()` 省略时不设总时限 |
| `file_account_scope` | `Option<String>` / `None` | 不含密钥的稳定 provider 账号身份，用于绑定或跨请求复用 provider 文件引用 |

### `CompletionRequest`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `model` | `String` | 配置中的显示名、wire ID、alias 或限定模型引用 |
| `web_search` | `Option<WebSearchConfig>` | `None` 为默认关闭；`Some` 启用所选连接的托管搜索 |
| `file_search` | `Option<FileSearchConfig>` | `None` 为默认关闭；Qwen Responses 的知识库搜索配置 |
| `previous_response_id` | `Option<ResponseId>` | Responses 续接的上一个响应 ID |
| `system` | `Vec<SystemBlock>` | 系统提示块：`text` 和 `cacheable` |
| `messages` | `Vec<ConversationMessage>` | 本轮输入及调用方维护的历史 |
| `tools` | `Vec<ToolSpec>` | 工具名称、说明、JSON Schema 与 `strict` 标志 |
| `tool_choice` | `ToolChoice` | `Auto`、`Any`、`None` 或 `Tool { name }` |
| `max_tokens` | `Option<u32>` | 输出 token 限制，由协议映射 |
| `temperature` | `Option<f32>` | 采样温度，provider 的支持范围由调用方确认 |
| `thinking` | `Option<ThinkingConfig>` | 思考模式、数值／动态预算及 effort，见[推理控制](inference.md) |
| `service_tier` | `Option<ServiceTier>` | Standard／Fast；不设置时保留服务端默认值 |
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
| `Video` | `source: VideoSource`；目前 MiniMax M3 通过已上传的 `video_understanding` 文件引用接收视频 |
| `ProviderContent` | `protocol`、`value`；保留 Responses reasoning、Chat reasoning 和 Claude/DeepSeek 搜索的原生内容，以便下一轮重放 |

Base64 源携带 `media_type` 和 `data`；URL 源携带 `url`。库不执行工具、不自动下载附件。工具调用返回后，宿主执行工具，把带相同 `tool_use_id` 的结果加入下一轮输入。

### `CompletionResponse`

`UsageReport` 将 `Option<Usage>` 与 `Missing`、`Partial`、`Complete`、`Invalid` 状态统一保存。仅有 usage 数值的旧格式不再接受；实际成本 API 只接受完整报告。

返回 `message: ConversationMessage`、`web_search: Option<WebSearchResult>`、`file_search: Option<FileSearchResult>`、`stop_reason: StopReason`、`usage: UsageReport`、`model: String`、`response_id: Option<ResponseId>`、`inference: InferenceReport` 和 `executed_profile: Option<String>`。高层 `complete()` 会设置实际成功的连接名称；直接调用 codec 解码时此字段为 `None`。`StopReason` 包括 `EndTurn`、`ToolUse`、`MaxTokens`、`StopSequence`、`Refusal` 和 `Other(String)`。

`inference` 保留请求的推理 effort 与服务档位、供应商回报的实际档位及本地执行时间。请求值不能证明实际档位，详见[推理控制与价格](inference.md)。

### 宿主工具执行与上下文恢复

以下函数写在调用方应用中。`append_tool_results` 接收 `response.message` 和应用自己的工具执行器，保留完整 assistant 消息（包括签名与原生内容），并用调用 ID 配对结果。宿主在回调中处理权限与工具错误，然后决定是否发起下一次请求。若 `response.executed_profile` 有值，后续重放应使用该实际执行连接。

此示例使用完整历史重放（`previous_response_id: None`）。状态续接则使用返回的响应 ID，仅发送新增输入，并固定到相同连接。下面的上下文恢复函数演示宿主“最多重试一次”的策略：调用方提供 `reduce`，它必须保留有效的工具调用／结果配对及重放签名；这不是客户端自动执行的行为。

```rust,no_run
use lingxi_llm_client::protocol::{
    CompletionRequest, CompletionResponse, ContentBlock, ConversationMessage,
    LlmError, MessageRole,
};
use lingxi_llm_client::{LlmClient, RequestOptions};

fn append_tool_results(
    request: &mut CompletionRequest,
    assistant: ConversationMessage,
    mut execute: impl FnMut(&str, &serde_json::Value) -> Result<String, String>,
) -> bool {
    let results: Vec<_> = assistant.tool_uses().map(|(id, name, input)| {
        let (content, is_error) = match execute(name, input) {
            Ok(output) => (output, false),
            Err(error) => (error, true),
        };
        ContentBlock::ToolResult {
            tool_use_id: id.clone(), content, is_error, blocks: None,
        }
    }).collect();
    request.messages.push(assistant);
    let has_results = !results.is_empty();
    if has_results {
        request.messages.push(ConversationMessage {
            role: MessageRole::User, content: results,
        });
    }
    has_results
}

async fn call_with_one_context_retry(
    client: &LlmClient,
    profile: &str,
    request: CompletionRequest,
    options: &RequestOptions,
    reduce: impl FnOnce(CompletionRequest, &LlmError) -> CompletionRequest,
) -> Result<CompletionResponse, LlmError> {
    match client.chat().complete_in(profile, &request, options).await {
        Err(error @ (LlmError::ContextOverflow { .. } | LlmError::RequestTooLarge { .. })) => {
            let reduced = reduce(request, &error);
            client.chat().complete_in(profile, &reduced, options).await
        }
        result => result,
    }
}
```

### 从共享 Agent API 迁移

客户端现在自行定义协议类型。将旧 `lingxi-agent-api` 导入改为 `lingxi_llm_client::protocol`；保留自有领域类型的应用需在边界转换，不提供 Rust 类型身份兼容层。

- 删除 `CompactTrigger` 和 `triggers_reactive_compaction()`。压缩状态留在宿主，按上述示例匹配通信错误。
- 删除 `ProviderProfile::vision_delegate` 和两个 `MediaDelegation*` 错误。媒体委派及其结果由宿主管理。配置和 Rust 字面量均需删除该字段。
- 从 `LlmError` 和 `LlmErrorKind` 删除 `OAuthRefreshDead`。宿主处理刷新失败；认证器向客户端报告请求认证失败时返回 `Authentication`。现有认证方式和认证失败转移继续可用。
- 删除未使用的 `TokenEstimate`、`TokenEstimateSource`。本地估算使用 `estimate_local_tokens()` / `estimate_local_tokens_in()` 及其 `LocalTokenEstimate` 返回值；供应商实际用量仍由 `Usage` 表示。

错误反序列化不再接受已删除的变体；保存过这些错误的宿主应将其迁入自身错误模型。provider 管理、账号用量查询、费用估算和本地 token 计数继续保留。

## Web Search 接口

`web_search()` 和 `web_search_stream()` 接受普通 `CompletionRequest`、本次搜索的 `WebSearchConfig` 和 `RequestOptions`。两种方法只克隆请求并设置 `web_search`，随后使用与 `complete()` / `stream()` 相同的路由、认证和故障转移逻辑；原请求不会被修改。传入的配置覆盖请求中已有的 `web_search`。也可以直接设置 `request.web_search = Some(config)` 后调用普通方法，效果相同。这里的“接口”是 Rust 客户端方法，本库不提供独立 HTTP 搜索服务。

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, ConversationMessage, LlmError, Secret, WebSearchConfig};
use lingxi_llm_client::{builtin_providers, LlmClientBuilder, RequestOptions};

async fn search(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let profiles = builtin_providers()?;
    let client = LlmClientBuilder::new(&profiles)?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
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
use lingxi_llm_client::protocol::{CompletionRequest, LlmError, StreamEvent, WebSearchConfig};
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

## Qwen 知识库 File Search

Qwen Responses profile 可在请求上设置单个知识库 ID 和其 Model Studio workspace ID。该功能目前适用于 Qwen Max / Flash 支持的 Responses 连接；内置的北京、新加坡、美国和香港 Qwen Search profile 已声明 `extra.file_search = "qwen"`。客户端会使用 workspace 专属的区域域名发送该 Responses 请求；普通模型请求仍使用 profile 配置的 base URL。

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, FileSearchConfig, Secret};
use lingxi_llm_client::{LlmClient, RequestOptions};

// client 必须使用 Region::ChinaMainland；国际连接请改用对应区域的 profile 和凭证。
async fn ask_qwen(client: &LlmClient, api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3.8-max",
        "messages": [{"role":"user","content":[{"type":"text","text":"按知识库回答问题"}]}]
    }))?;
    request.file_search = Some(FileSearchConfig {
        knowledge_base_id: "kb-123".into(),
        workspace_id: "ws-example".into(),
    });
    let options = RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..RequestOptions::default()
    };
    let response = client.chat().complete_in("qwen-search", &request, &options).await?;
    if let Some(search) = response.file_search {
        for hit in search.hits {
            println!("{}: {}", hit.filename.unwrap_or_default(), hit.text.unwrap_or_default());
        }
    }
    Ok(())
}
```

`file_search` 与 `web_search` 可以在同一请求同时启用。流式调用会收到 `StreamEvent::FileSearch { result }`，并在 Responses 完成事件重复附带搜索结果时去重。此功能是 Qwen 托管知识库查询，不是客户端文件上传；Qwen-Long 的文档上传和 `fileid://` 引用见[文件附件指南](file-attachments.zh.md)。

## 流式响应

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, LlmError, StreamEvent};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn read_stream(
    client: &LlmClient,
    request: &CompletionRequest,
    options: &RequestOptions,
) -> Result<(), LlmError> {
    let mut stream = client.chat().stream(request, options).await?;
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
| `ProviderContent { block, protocol, value }` | 完整原生 reasoning 数据；按输出索引存为 `ContentBlock::ProviderContent` 并在下一轮原样回传。同索引的思考增量仅用于展示，不能替代该数据 |
| `ThoughtSignature { block, signature }` | 思考签名，需与对应 block 一起保留 |
| `RedactedThinking { block, data }` | 不透明思考数据 |
| `ToolCallDelta { block, id, name, arguments_fragment }` | 工具参数片段；按 block 累积后再解析 JSON |
| `WebSearch { result }` | 托管搜索来源及 provider 原生元数据；可能出现多次 |
| `FileSearch { result }` | Qwen 托管知识库查询与命中结果；重复的最终输出会去重 |
| `Inference { report }` | 服务端档位与 effort 观测；与请求值区分 |
| `End { stop_reason, usage, inference }` | 解码器输出的结束事件 |

Chat Completions 中的 OpenRouter 风格 `reasoning` / `reasoning_details` 使用 `ProviderContent { protocol: OpenAiChat, value: {"type":"chat_reasoning", ...} }` 保存消息级回放数据。普通响应会保留该块；流式响应在正常终态前发送一次完整块。宿主应在对应 assistant 消息中保留一份该块，思考增量仅用于展示。编码器按原始顺序回传 `reasoning_details`，拒绝重复 envelope、非 assistant 角色和跨协议重放。协议字段 `reasoning_content` 由 `preserve_reasoning_content` 控制。

`observed_usage()` 可返回部分计数；`usage_is_complete()` 才表明计数完整且自洽，不能仅凭存在 `End` 就认定可用于计费。如果底层提前 EOF、尚未观察到 provider 的终止标记，解码器返回 `StreamInterrupted`。读取错误后应结束本轮；客户端立即释放底层响应和待处理帧，保留句柄仍可读取已观测用量与响应头。`ModelStream` 返回给调用方之后的内容错误不会自动切换连接，避免重复生成或重复执行工具。

Anthropic 的 `message_start` 计数属于初始值，即使包含数值有效的 `output_tokens` 也不视为最终用量；需要后续最终用量报告。`RedactedThinking` 的不透明数据应按 block 保存并原样放回下一轮消息。Gemini 提示词拦截和 Responses 拒绝内容会保留 `StopReason::Refusal`，完整响应与流式响应采用一致语义；截断或 provider 错误仍保留自己的终态。

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
| `pricing` | `billingMode`、可选 `peak` |
| `signing` | 可选 `region`、`service`、`project`，供宿主签名实现使用 |
| `azure` | 可选 `api_version` 和 `deployment` |
| `info` | 展示名、说明、控制台/API key/文档链接和凭证提示 |
| `extra` | 协议选项、额外 body/header 等 JSON 数据 |
| `supports_websockets` 等 WebSocket 字段 | 配置元数据；高层客户端当前仍使用 HTTP |

`ModelProfile` 必须指定 `display_model`、`request_model`、`billing_model`；可添加 `aliases`、`description`、`metadata`、`capability_support`、`pricing` 和模型级 `billing_mode`。三种模型名分别用于显示/匹配、请求协议和价格归属。`metadata` 保存上下文窗口、输出上限、模态等目录数据。

`capability_support` 使用 `unknown`、`supported`、`unsupported` 三种状态。`ModelProfile::capability_support_for(ModelCapability)` 返回明确的支持状态；缺失字段保持未知。旧布尔能力字段和回退逻辑已删除。推理参数的范围和组合约束另见 `info.features`。

`"capability_support": {"tools": "unsupported"}` 只声明工具调用不受支持，其余能力保持未知。

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

Azure 必须配置 `azure.api_version`；`azure.deployment` 省略时使用 `request_model`。Vertex Claude 请求体使用平台版本 `vertex-2023-10-16`；Bedrock 的模型 ID/ARN 按单个路径参数编码，流式模式由路径决定，请求体不包含 `stream`。托管平台 codec 负责 URL 和 wire body，不自动获取云凭证或实现 AWS SigV4 签名。

### 解析和故障转移

`resolve()` 支持完整 `display_model`、`request_model`、alias，以及 `profile/model`、`group/model` 限定引用。合法的含 `/` 原生模型 ID 保持完整；如果同一个字符串同时表示一个原生 ID 和另一个目标的限定引用，则返回歧义错误，不发送请求。限定名同时是连接名和组名时，连接名优先。

已知凭证所属 profile 时，推荐使用 `complete_in(profile, request, options)`、`stream_in()` 或相应的搜索方法。把原生模型 ID 放在 `request.model` 中，独立传入 profile，避免把目标连接和模型 ID 拼成字符串。例如同时启用 OpenAI、OpenRouter 时，调用 `complete_in("openai", ...)` 并设置 `model: "gpt-4o"` 会选择 OpenAI；调用 `complete_in("openrouter", ...)` 并设置 `model: "openai/gpt-4o"` 会选择 OpenRouter。未限定的 `openai/gpt-4o` 因两种解释冲突而报错。

`resolve_in(model, Some(name))` 用于预先查看相同的限定路由；执行时使用对应的 `_in` 方法。主凭证必须属于解析后的起始连接，备用连接仍只能使用 `fallback_credentials` 中对应 profile 的凭证。

跨组同名返回 `ResolveError::AmbiguousAcrossGroups`；原生名称与限定引用指向不同目标时返回 `AmbiguousNativeAndQualified`；单连接中多个模型匹配返回 `DuplicateOnProfile`；没有匹配返回 `UnknownModel`。请求方法会将解析错误转换为 `LlmError::ModelUnavailable`。

`ResolvedRoute` 包含 `provider_id`、`profile_name`、`request_model`、`display_model`、`pricing_model`、`capability_support`、`connection_chain` 和 `failover`。不要把任意手工构造的 route 当作客户端已经验证过的路由。

`connection.group` 省略时，组名为 `profile_name`。连接按 `(order, profile_name)` 排序。备选连接必须在同组、提供相同 `request_model` 且有效计费模式相同；若同一备选连接有多个满足条件的模型行，则跳过该连接，避免猜测行级覆盖。限定起始连接不禁用该组的故障转移。未指定连接时优先选择可见连接；显式指定 profile 仍可选中隐藏连接，隐藏连接也可参与故障转移。

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

该片段需合并进完整 `ProviderProfile`。`rateLimit` 匹配限流和额度耗尽，`network` 匹配传输错误和超时，也包括内置 HTTP 客户端归类为传输错误的 TLS 失败；上下文超限和不支持能力不触发切换。策略取自起始 route，每个连接至多尝试一次；库不执行退避等待或同连接重试。缓冲与流式请求一旦收到非成功 HTTP 状态，即使错误体读取中断或超时，也保留状态和响应头以及已读取的最多 64 KiB 错误体，由 codec 分类后决定故障转移。成功流交给调用方后发生的中断不会自动重放请求。

### 内置配置与扩展参数

`builtin_providers() -> Result<Vec<ProviderProfile>, PresetError>` 返回编译进库的静态目录。`merge_providers(user)` 保留用户条目，并追加未被同名用户条目覆盖的预设；这是整条 profile 覆盖，不是字段合并。用户输入中的重复名称仍由 builder 拒绝。`PresetError` 分为解析失败 `Invalid` 和空模型列表 `NoModels`。

### 本地保存与多账号

`LlmClient` 创建后可调用 `set_config_dir(path)`。库在该目录管理 `providers.json`，立即加载其中的 profile，并将目录固定为绝对路径，后续工作目录变化不会改变保存位置；同名账户的连接设置覆盖静态定义，模型字段按来源合并，详见 [v2 配置说明](architecture-migration.md)。再次设置目录时会清除上一个目录加载的配置。多个客户端写入同一目录时，库使用 `.providers.json.lock` 协调写入，并在锁内读取最新配置后应用本次修改；直接修改 JSON 的外部程序也需要遵守这个锁。`sync_provider(profile_name, credential).await` 使用该连接自己的凭证读取模型目录，按 `request_model` 更新现有模型、加入新模型，再保存配置。目录缺失、请求失败或写盘失败时，旧文件与内存配置保持不变。同步不会删除目录未返回的旧模型，也不会改变已有模型的 `hidden`、价格、能力或别名。目录明确报告不兼容的模型时，同步会按 profile 持久保存排除记录，并从有效模型列表和路由中排除该模型，避免旧配置、白名单变化或重启绕过排除；仍在白名单中的完整模型元数据继续保存，目录明确恢复兼容后复用原有设置；目录未返回的 ID 不会被视为不兼容。后续目录明确报告支持该 ID，或显式替换 profile、恢复内置 profile、删除 profile 时，会清除相应的排除记录。Gemini 行缺少 `supportedGenerationMethods` 时支持状态仍为未知，会保留模型行但不会清除已有排除记录。

同步过程的文件读取、文件锁和写盘运行在 Tokio 的 blocking 线程池上。`sync_provider()` 是串行便捷入口；需要在目录网络请求期间继续使用客户端时，先调用同步方法 `prepare_provider_sync()` 获得独立的 `ProviderSyncOperation`，再执行其 `fetch().await`，最后通过 `apply_provider_sync(result).await` 提交。准备操作不访问磁盘，并复制本次凭证和必要配置；operation 不借用客户端，宿主可在网络等待前释放自己的客户端锁。

```rust,no_run
use lingxi_llm_client::{LlmClient, ProviderStoreError};
use lingxi_llm_client::protocol::Secret;

async fn refresh(
    client: &mut LlmClient,
    credential: &Secret<String>,
) -> Result<usize, ProviderStoreError> {
    let operation = client.prepare_provider_sync("primary", Some(credential))?;
    // operation 已拥有获取目录所需的数据，可独立调度；不再借用 client。
    let result = operation.fetch().await?;
    client.apply_provider_sync(result).await
}
```

获取目录前会核对磁盘上的连接配置，提交时再次在文件锁内校验并合并最新配置。结果还绑定准备时的配置目录代次；切换目录后，即使再切回原路径，旧结果也会被拒绝。丢弃尚未提交的获取操作不会修改配置。提交操作在进入写入前检查取消状态，但写入开始后取消仍可能留下已更新的文件和未更新的客户端快照；遇到这种取消时，再次调用 `set_config_dir()` 可重新加载已持久化状态。同步配置管理方法（如 `add_provider()`）仍是阻塞接口，宿主从异步任务调用时应安排在合适的阻塞执行环境中。

目录观察没有版本号；同一 profile 的并发获取结果按 `apply_provider_sync()` 的提交顺序应用。新鲜度校验只验证连接配置和配置目录代次。若宿主要求最近发起的刷新优先，应串行刷新该 profile，或在提交前丢弃过期结果。

| 操作 | API | 保存行为 |
| --- | --- | --- |
| 新增 / 修改 | `add_provider(profile)` | 按 `profile_name` 新增或整条替换并写盘；也可用于多账号中的单个账号 |
| 查询 | `provider(profile_name)`、`profiles()`、`providers()`、`deleted_builtin_profiles()` | 读取配置、列表摘要及已软删除内置项的名称 |
| 删除 | `remove_provider(profile_name)` | 自定义 profile 从文件删除；内置 profile 记录软删除，重启后仍停用 |
| 恢复内置项 | `restore_builtin(profile_name)` | 清除软删除标记并保存库内预设；即使 builder 有同名自定义覆盖也以预设为准 |

删除的是单个连接账号，不会连带删除同组的其他账号；如需停用整个 provider，逐个删除其 profile。重加一个已软删除的同名配置也会清除软删除标记。自定义 profile 若由宿主在每次启动时再次传给 builder，宿主还须从自己的输入中移除它。

`set_tracked_models(provider_id, request_model_ids)` 为同一家 provider 的所有账号设置全局模型白名单，并保存在 `providers.json`。未配置白名单的 provider 不显示或保存任何模型；目录同步只合并白名单中的模型，保存时也只写入白名单中的模型。先调用 `add_provider` 再设置白名单也能使用本次会话提供的模型；重启后若要扩大白名单，需要由 builder 再次提供这些模型或重新同步目录。如果其他客户端修改了同名 profile 的模型列表，本客户端不会再用旧缓存补回被删除的模型。`tracked_models(provider_id)` 可读取当前名单；`untrack_model(provider_id, request_model)` 从名单删除单个模型，再次同步也不会重新加入。白名单决定是否跟踪、保存和显示；模型级 `hidden` 只决定已跟踪模型是否出现在 `models()` 中。

同一家 provider 的每个账号使用独立的 `profile_name` 和 `connection.connection_id`，并设置相同的 `connection.group`。主账号设置 `hidden: false`，备用账号设置 `hidden: true`，按 `order` 排序。`sync_provider` 逐账号执行，模型目录不在账号间复制。故障转移需要在 `RequestOptions.fallback_credentials` 中按备用 `profile_name` 提供其密钥；配置文件仅保存 `env` 或 `host_managed` 等凭证来源，不保存明文密钥。`CredentialConfig::Static` 会被保存接口拒绝。读取和保存配置时，已知认证头（包括 `x-goog-api-key`）及 `extra.credential_header` 指定的头都不能放入 `extra.headers`；凭证应通过请求选项传入。

```rust,no_run
use lingxi_llm_client::protocol::{ProviderProfile, Secret};
use lingxi_llm_client::LlmClientBuilder;

# async fn example(primary: ProviderProfile, spare: ProviderProfile) -> Result<(), Box<dyn std::error::Error>> {
let mut client = LlmClientBuilder::new(&[])?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
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

`ProviderStoreError` 区分文件、JSON、配置验证、目录请求、分页和阻塞任务错误。同步最多读取 100 页，并拒绝重复 cursor；若请求期间连接配置被其他客户端修改，同步返回 `ProfileChanged` 且不写入旧连接的模型。启动后如需使用已保存的 provider，须再次调用 `set_config_dir`；本库不自动读取环境变量中的凭证。

`extra.body` 用于补充协议未写入的 JSON 字段，不能覆盖已写字段、模型身份、续接 ID 或 `complete()` / `stream()` 选择的响应模式；`extra.headers` 用于非凭证附加头，不能覆盖 codec 已写头。认证头应交由认证器。

OpenAI Chat 连接可通过 `extra.max_tokens_field` 选择输出上限字段：默认 `"max_tokens"`，直接调用要求新字段的 OpenAI 模型时配置 `"max_completion_tokens"`。该配置在 `extra` 顶层，不在 `extra.body` 内；其他值或非字符串会返回 `InvalidRequest`。Chat 文件输入不支持 `DocumentSource::Url`，会在发送前返回 `UnsupportedCapability`；使用 Base64 内容或支持 URL 的其他协议。

当推理 token 独立于 completion token 计数时，OpenAI Chat 配置可设置 `extra.reasoning_tokens_separate = true`。内置 Grok Chat 已启用；客户端会在普通响应和流式响应中将推理量归入输出总量。其他配置保持默认的子集口径。直接使用 codec 时，应调用 `decode_response(response, &context)` 和 `stream_decoder(&context)`，以应用该连接的计数约定。

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
| 重定向 | 默认禁止跟随；`send` 保留原始重定向响应 |
| 重试 | 不自动重试；高层显式配置的故障转移仍可生效 |
| 连接超时 | 30 秒 |
| 流读取空闲超时 | 默认 60 秒；`HttpTransport::with_read_timeout(Duration)` 可在客户端级调整 |
| 总超时 | `complete()` 默认 120 秒（含视频的请求默认两小时）；`stream()` 默认不设总时限，活跃长流可持续运行；`RequestOptions.total_timeout` 为两者指定总时限，包含流读取 |
| HTTP 错误状态 | 保留状态、响应头和 body，由 codec 分类，不提前丢弃错误体 |
| 网络错误 | 返回语义化 `LlmError`，错误消息不包含请求 URL、认证头或 body |

复用客户端可复用连接池；丢弃响应流会释放对应资源。流式请求的总时限不会因收到新 chunk 而重置。需要不同网络策略时，可继续注入自定义 `Transport`。

`HttpExecutor` 统一收集正文、限制大小并执行单调时钟 deadline，即使自定义 transport 忽略 timeout 仍然生效。认证、上传、轮询及流读取共用请求预算。codec 创建自己的 SSE 或 EventStream decoder，`ModelStream` 不再根据 Content-Type 推测分帧方式。

### 自定义传输与时钟

`LlmClientBuilder::with_transport(http, &profiles)` 接收 `Arc<dyn Transport>`，同时使用内置 `SystemClock`。`Clock::now() -> SystemTime` 用于价格时段计算；测试可通过构建器的 `with_clock(Arc<dyn Clock>)` 覆盖系统时钟。普通使用无需实现或导入 `Clock`。

`Transport: Send + Sync + 'static` 使用 `async_trait`，需要实现：

| 方法 | 返回类型 | 责任 |
| --- | --- | --- |
| `send(HttpRequest)` | `Result<StreamResponse, LlmError>` | 返回原始字节流、状态和响应头；禁止自动重定向和自动重试 |

`HttpRequest` 包含 `method`、`url`、`headers: Vec<(String, String)>`、`body: Bytes`、`timeout: Option<Duration>`。`HttpResponse` 包含 `status: u16`、`headers`、`body`，并提供不区分大小写的 `header()`。`StreamResponse` 也提供 `header()`。

传输实现应负责 TLS、代理、连接池、超时、网络错误归类、响应资源释放和重定向策略。内置传输对非 2xx 响应最多读取 64 KiB 错误体后交给 codec 分类；达到上限、连接中断或错误体读取超时只会截短 body，不会丢弃已收到的状态和响应头。成功状态的响应体必须完整读取，读取中断或超时会返回错误。不得把错误 HTTP 状态预先丢弃为不含响应体的通用网络错误。


## 模型目录

内置 `OpenAiChatDirectory`、`AnthropicMessagesDirectory` 和 `GeminiDirectory`。不存在对应 reader 或配置 `model_list = "none"` 时，`directory_for()` 返回 `None`，并不表示现有模型不可调用。Responses 端点若使用 OpenAI models 列表，应显式配置 `model_list = "open_ai_chat"`。

目录操作是显式的：构造请求 → 认证 → 传输 → 解码 → 使用 cursor 获取下一页。以下示例适用于传入 API key 的 profile；其他认证策略应传入匹配的认证器。

```rust,no_run
use lingxi_llm_client::protocol::{LlmError, ProviderProfile, Secret};
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
        let response = lingxi_llm_client::HttpExecutor::new(http).execute(request).await?;
        let page = directory.decode_page(&response)?;
        models.extend(page.models);
        cursor = page.next_cursor;
        if cursor.is_none() {
            return Ok(models);
        }
    }
}
```

生产宿主还应设置总刷新时限/页数限制，避免异常服务反复返回相同 cursor。内置目录请求单页超时为 30 秒，由公共执行器约束。

`ModelPage { models, next_cursor }` 的 cursor 是不透明字符串，原样传回。Anthropic 的 `has_more` 必须是布尔值；为 `true` 时还必须提供非空字符串 `last_id`，缺失、类型错误或矛盾的分页信息会返回解析错误。这与其[模型列表响应结构](https://platform.claude.com/docs/en/api/models/list)一致。Gemini 的 `nextPageToken` 可缺省、为 `null` 或空字符串来表示最后一页；其他非字符串值会返回解析错误。若 Gemini 行包含 `supportedGenerationMethods`，该字段必须是字符串数组；明确不含 `generateContent` 的行会从可调用模型中排除，明确包含该操作的行会确认兼容。缺少该字段时，支持状态仍为未知，行可用于尚无排除记录的 profile，但不会清除已有排除记录。`LiveModel` 包含 `request_model`、可选 `display_name`、`description`、`context_window` 和 `max_output_tokens`，不包含价格。直接读取目录不会自动加入配置；可使用前述 `sync_provider()` 或分阶段同步接口合并已跟踪模型，也可由宿主自行管理目录。

## 用量与费用

`Usage` 的 `input_tokens`（未缓存输入）、`output_tokens`、`cache_read_tokens` 和 `cache_write_tokens` 是互斥计费桶。`reasoning_tokens` 已包含在输出 token 内，不能再相加。`Usage::total()` 求四个计费桶之和。`cache_write_1h_tokens` 是 `cache_write_tokens` 中一小时 TTL 写入的子集，不能再次加进总量；缺省为 0。费率结构通过 `cache_write_1h_per_million` 单独表示一小时写入价格；缺少该费率且实际产生一小时写入时返回 `CostUnavailable`。仅提供聚合缓存计数、未报告 TTL 明细的响应继续使用配置的通用缓存写入费率。Gemini 工具输入 token 计入输入桶，思考 token 计入输出桶。

`Usage.cost: Option<ReportedCost>` 是 provider 实际报告的金额。`ReportedCost.nano_usd` 以十亿分之一美元为单位；`from_usd(f64)` 拒绝负数、非有限数和溢出值。金额缺失表示未知，不是免费。

`Usage.server_tool_usage` 单独保留 provider 报告的托管工具计数，目前包括 Web Search 和 File Search 请求数；这些请求数不属于 token 总量，token 计费或总量计算不会把它们相加。

```rust,no_run
use lingxi_llm_client::protocol::{LlmError, PricingContext, Usage};
use lingxi_llm_client::LlmClient;

fn show_estimate(client: &LlmClient, model: &str, usage: &Usage) -> Result<(), LlmError> {
    let route = client.resolve(model)?;
    let cost = client.estimate_cost(&route, usage, &PricingContext::default())?;
    println!("estimated {}: {}", cost.currency, cost.total_cost);
    Ok(())
}
```

`CostEstimate` 从 `lingxi_llm_client::client::pricing` 导入，包括 `pricing_model`、`submission`、五个费用字段（input/output/cache_read/cache_write/reasoning）、`total_cost` 和价格 `source`。

- `TokenPricing` 每项费率以 `currency` / 百万 token 为单位，未指定币种时为 USD；非零用量缺少对应费率时返回 `CostUnavailable`。使用中的费率必须非负且有限，费用计算溢出也返回 `CostUnavailable`。
- 缺少价格或关键计费条件时统一返回 `CostUnavailable`；用 `price_quote` 查询价格状态。
- `Submission::Batch` 使用各桶显式的批处理费率，不假定固定折扣，也不表示客户端会提交 batch 作业。
- 如果单独配置 reasoning 价格，则从输出桶中拆出 reasoning，避免重复计费；`reasoning_tokens > output_tokens` 返回 `CostUnavailable`。
- `PeakSchedule` 使用 UTC 时间窗（`HH:MM-HH:MM`，结束时间可为 `24:00`）和可选工作日约束；使用 `PricingContext.unix_seconds` 查询指定时刻。所有公开估价入口共用价格规则选择器；`TokenPricing::at` 和公开的 `pricing::estimate` 已移除。

费用是目录估算，不替代 provider 账单。`estimate_cost` 只用于请求前对初始 route 预估；故障转移后应使用 `estimate_actual_cost(&route, &response, submission)`。流式响应使用 `estimate_stream_cost(&route, &stream, submission)`，同时保留实际连接与服务档位。两种实际连接估价仍使用目录价格，可能与 provider 账单不同。

### 本地输入 Token 估算

```rust,no_run
use lingxi_llm_client::{
    protocol::CompletionRequest, LocalTokenCountError, LocalTokenEstimate, LlmClient,
};

fn estimate_input(
    client: &LlmClient,
    request: &CompletionRequest,
) -> Result<LocalTokenEstimate, LocalTokenCountError> {
    let estimate = client.estimate_local_tokens(request)?;
    println!("{} tokens via {}", estimate.input_tokens, estimate.tokenizer);
    if estimate.is_partial {
        println!("not counted: {:?}", estimate.uncounted_components);
    }
    Ok(estimate)
}
```

这两个方法同步执行，不发送网络请求、不读取凭证、不解析远程 URL，也不调用附件解析器。`estimate_local_tokens` 在客户端当前区域解析模型并使用首选 route；`estimate_local_tokens_in(profile_or_group, request)` 将解析限制到指定连接或连接组。故障转移链只估算首选连接的模型，不会预估之后可能接手的连接。

估算器只计输入：system 与对话中的文本、文本型文档、工具名／描述／JSON schema、工具调用和文本结果。结构化内容按紧凑 JSON 计数；消息边界、工具封装和 provider 请求格式以固定开销近似。`max_tokens` 是输出上限，不计入输入 token；也不会把温度等生成参数当作 prompt 文本。`LocalTokenEstimate.is_estimate` 始终为 `true`，因为不同 provider 的消息封装和隐藏模板不完全相同。估算值用于本地预估，provider 响应中的 `Usage` 才是实际用量依据。

默认构建不包含 tokenizer 后端或嵌入资产。按需开启 `tokenizer-openai`、`tokenizer-deepseek`、`tokenizer-qwen`、`tokenizer-kimi`、`tokenizer-glm`，或聚合 feature `tokenizers-all`。已知模型未启用后端时返回 `FeatureDisabled`；未知模型返回 `UnsupportedModel`。

当前只对能与打包 tokenizer 资产明确对应的模型计数：OpenAI 使用 `tiktoken-rs` 的模型映射；DeepSeek `deepseek-v4-pro`、`deepseek-flash`；Qwen `qwen3.8-flash`、`qwen3.8-max`；Kimi `kimi-k3`；Z.AI/Zhipu `glm-5`。provider 与模型 ID 必须同时匹配；其他模型、Anthropic、Gemini、xAI、OpenRouter 以及 MiniMax M3 返回 `LocalTokenCountError::UnsupportedModel`。其中 Anthropic、Gemini 和 xAI 的官方 token 计数依赖线上接口，OpenRouter 依赖上游用量；未知模型不会回退到字符比例估算。

遇到本地无法取得或无法按文本 tokenizer 计数的内容时，仍返回可见文本的估值，同时设置 `is_partial = true` 并在 `uncounted_components` 中列明：图片、非文本／远程文档、视频、provider 文件、托管 Web Search／File Search 内容、Responses 前序服务端状态、provider 专属结构化块、签名及 provider metadata。每次出现都会单独列出。加密思考数据不按明文分词：Anthropic 协议及其托管变体会将其列为 opaque 内容遗漏，其他内置协议会跳过不发送的该块。文本型文档的文本会计入；远程资源不会被下载，附件也不会被读取。可用 tokenizer 与模型映射、来源、许可证和 SHA-256 记录在 [`data/tokenizers/README.md`](../data/tokenizers/README.md)。源码归档包含 XZ 压缩资产及许可证，各 feature 仅嵌入对应资产；MiniMax M3 资产因其上游非商业许可限制不打包。

## 账户额度与账户 Token 用量

`AccountQuery.execution` 默认每账户总预算 60 秒、每次 HTTP/RPC 30 秒。总预算从取得批量执行槽位开始，分页和后续调用不重置预算。批量默认并发 4，可通过 `with_account_concurrency(NonZeroUsize)` 调整；完成的查询立即释放槽位，最终结果仍按 profile 原顺序返回。`AccountFailure::Timeout` 只标记尚未完成字段，已写入指标保留。自定义来源实现 `fetch(&AccountFetchContext, &mut AccountReport)`，每取得一个字段就写入报告，再等待下一次操作。

`LlmClient::account_usage(profile_name, &AccountQuery)` 只读查询一个连接的账户状态；`accounts_usage(&BTreeMap<String, AccountQuery>)` 逐一查询所有已配置连接（包括隐藏连接），逐连接返回结果。缺少某个连接的查询参数时，该连接返回 `AccountUsageError::MissingQuery`，不会影响其他连接。`AccountQuery::new(AccountIdentity::ApiKey | AccountIdentity::AuthUser)` 显式标明账户身份；不要从 `ProviderProfile.auth` 推断，因为普通 API Key 也可能使用 Bearer 请求头。查询时间默认为最近 30 天，使用 UTC Unix 秒，可通过 `since_unix`、`until_unix` 修改。

```rust,no_run
use lingxi_llm_client::{AccountIdentity, AccountMetric, AccountQuery, LlmClient};
use lingxi_llm_client::protocol::Secret;

async fn show_balance(client: &LlmClient, key: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new(key));
    let account = client.account_usage("deepseek", &query).await?;
    if let AccountMetric::Available { scope, value, .. } = account.balance {
        println!("scope: {:?}, balances: {:?}", scope, value);
    }
    Ok(())
}
```

`AccountSnapshot` 的 `balance`、`token_usage`、`cost_usage`、`quota_windows`、`subscription` 各自是 `AccountMetric`：可以分别为有数据、未公开、缺少凭证、供应商未报告或查询失败。有数据时携带实际统计范围（Key、用户、项目、团队、组织等）；组织级数据不会伪装成某个连接独有用量。`AccountTokenBucket.input_tokens` 包含供应商报告的缓存输入，`cached_input_tokens` 和 `cache_write_tokens` 是其子集；它与单次请求的 `Usage` 计费桶不同。余额与历史费用以十进制字符串输出，费用金额使用主货币单位。额度窗口提供名称、时长、使用／剩余百分比、重置时间及供应商明确给出的绝对数量；金额限额使用 `limit_decimal`、`remaining_decimal`，缺失项保持 `None`。只有官方账户权益能证实套餐时，订阅状态才是 `VerifiedActive` 或 `VerifiedInactive`。

普通 Key 可查询 DeepSeek、Moonshot 余额和 OpenRouter Key 消费上限。DeepSeek 的 `AccountBalance.is_available` 保留供应商报告的可用状态；即使余额非零，也可能暂不可使用。Key 消费上限不等于账户余额；OpenRouter 账户总额度需要单独的 Management Key。OpenAI、Anthropic、xAI 的账户用量或余额需要相应 Admin/Management 凭证；Google 项目用量需要 Cloud Monitoring 权限和项目 ID。将这些凭证放入 `management_credential`，所需的 `api_key_id`、`team_id`、`project_id` 等放入 `AccountQuery.selector`。管理请求只使用固定的官方地址，不读取模型连接的自定义 `base_url`。没有官方账户接口的字段返回 `Unsupported`，不会用一次模型响应的计数或目录价格冒充账户账单。

Qwen 的 `quota_windows` 需要该区域的 Qwen API Key 和 `selector.workspace_id`；`model_limit` / `workspace_limit` 是速率或使用量上限，不包含已消耗数量，因此 `used`、`remaining` 和百分比保持缺省。Qwen 历史成本使用单独的 `AlibabaAccessKey { id, secret, security_token }` 调用已签名的 GetBillingTrend；还需 `selector.api_key_id`，结果按日、模型和该区域返回。AccessKey 仅存在于该次 `AccountQuery`，客户端不保存或记录它。

```rust
use lingxi_llm_client::{AccountIdentity, AccountQuery, AlibabaAccessKey};
use lingxi_llm_client::protocol::Secret;

let mut query = AccountQuery::new(AccountIdentity::ApiKey);
query.credential = Some(Secret::new("qwen-api-key".into()));
query.selector.workspace_id = Some("ws-example".into());
query.selector.api_key_id = Some("key-id-from-aliyun".into());
query.alibaba_access_key = Some(AlibabaAccessKey {
    id: "ram-access-key-id".into(),
    secret: Secret::new("ram-access-key-secret".into()),
    security_token: None,
});
```

MiniMax Token Plan 的 `quota_windows` 使用 `AccountQuery.credential` 查询 `/v1/token_plan/remains`。返回中的 `current_interval_usage_count` / `current_weekly_usage_count` 语义不明确时不会当作已用量或剩余量；只使用明确的剩余数量或百分比。MiniMax 按量付费余额和历史账单没有在这里假造为支持项，仍返回 `Unsupported`。

ChatGPT/Codex、GitHub Copilot 和 Kimi Code 的用户账户查询由宿主提供已经登录的官方本地服务或 SDK。单账号可通过 `register_account_source(provider_id, AccountIdentity::AuthUser, source)` 接入；同一供应商有多个 Auth 用户时，应通过 `register_profile_account_source(profile_name, AccountIdentity::AuthUser, source)` 分别绑定登录会话。共用单会话数据源会返回 `AmbiguousAccountSource`，避免额度归属错误。若提供 `AccountQuery.credential`，Copilot 会将它作为用户的 `gitHubToken` 传给 `account.getQuota`。`CodexAccountSource` 和 `CopilotAccountSource` 接受宿主实现的 `AccountRpc`，其 `call` 返回 JSON-RPC 的 `result` 内容，并将 RPC 错误转换成 `AccountFailure`；`KimiCodeAccountSource` 使用每次查询显式提供的回环地址及令牌。客户端不执行 OAuth 登录、刷新或凭证持久化。Codex 的 5 小时／每周窗口、Kimi Code 的 5 小时／可用时的 7 天窗口按实际响应返回；Copilot 仅返回官方 SDK 报告的窗口。Codex 的每日 Token 记录按 UTC 整天返回；时间区间只筛选与之相交的完整日桶，不按小时分摊。Kimi Code 的官方 `userinfo` 只给出等级名称，没有明确的订阅状态语义，因此其订阅状态保持 `Unknown`；GLM Coding Plan 暂无已确认的公开账户查询接口，也不会根据连接配置宣称已订阅。

Copilot 也可在每次查询的 `AccountQuery.credential` 中提供对应用户的 GitHub 令牌；此时共享 SDK 数据源会按令牌分别查询，不需要逐连接绑定。

替换、删除或重载已变更的连接会清除其按连接绑定的账户源；切换配置目录也会清除这些绑定。供应商级单账号会话源的隐式绑定同时失效，此后需要会话绑定的查询会返回 `AmbiguousAccountSource`；无状态来源及每次显式提供用户 token 的查询仍可复用。重新登录后应调用 `LlmClient::register_profile_account_source`，避免把旧会话额度标到新连接。

## 错误处理

`LlmError` 定义在客户端的 `protocol` 模块，`kind()` 返回可用于分类表的 `LlmErrorKind`。不要依赖 `message` 文本匹配控制流程。

| 变体 | 含义 / 宿主处理方向 |
| --- | --- |
| `Authentication`、`PermissionDenied` | 检查凭证、刷新或权限 |
| `InvalidRequest`、`UnsupportedCapability` | 调整请求或配置 |
| `RateLimited { retry_after, .. }`、`QuotaExceeded` | 限流/额度；退避或展示限制 |
| `ContextOverflow { limit, actual, .. }`、`RequestTooLarge` | 缩减历史或请求体 |
| `ModelUnavailable` | 模型无法解析或不可用 |
| `ProviderInternal`、`Overloaded` | provider 内部故障/过载 |
| `Transport`、`TransportTimeout`、`TlsCert` | 网络、超时、TLS 问题 |
| `StreamInterrupted` | 流内容损坏或意外中断 |
| `ProviderFileProcessing { file, .. }` | 文件就绪状态未确认；保存账号绑定的引用以便稍后继续轮询 |
| `CostUnavailable` | 价格数据不足 |

上述变体除特别列出的附加字段外均含 `message: String`。`retry_after` 是可选 `Duration`；当前内置解析器支持秒数形式的 `Retry-After`。宿主直接匹配 `ContextOverflow` 和 `RequestTooLarge`，自主决定是否缩减输入和再次调用；客户端不会因这些错误自动压缩或重试。

## 扩展接口

可选的 `directory::DecodedModelPage` 新增兼容性元数据，包含 `page: ModelPage`、`incompatible_model_ids` 和 `explicitly_compatible_model_ids`；`LiveModel.inference_features` 返回目录明确提供的推理控制信息。`ModelDirectory::decode_page_with_exclusions()` 默认调用原有 `decode_page()` 并返回两个空 ID 集合。

| 接口 | 必须实现的方法 | 注册 / 使用位置 |
| --- | --- | --- |
| `WireCodec` | `family`、`encode_request`、`encoded_body_len`、`decode_response`、`stream_decoder` | builder 的 `register_codec`；同协议族后注册覆盖前者 |
| `StreamDecoder` | `push_bytes`、`finish`、`usage_report` | codec 每个响应创建独立有状态 decoder |
| `Authenticator` | 异步 `apply(&mut HttpRequest, &ProviderProfile, Option<&Secret<String>>)` | builder 按认证策略注册，编码完成后调用 |
| `ModelDirectory` | `shape`、`list_request`、`decode_page`；可选 `decode_page_with_exclusions -> directory::DecodedModelPage` | builder 按目录 shape 注册；`DecodedModelPage` 包含原有 `ModelPage`、明确不兼容 ID 和明确兼容 ID；默认方法保留旧解码行为并返回空集合 |

`ProtocolFamily` 是封闭枚举；现有兼容协议的新 provider 只需配置，新增协议族需要修改枚举与 codec。codec 应把非成功响应转换为语义化 `LlmError`，归一化 token 用量，并保留思考签名和工具 ID。

`framing::sse::SseFrameSplitter` 提供 `new()`、`push(&[u8])` 和 `finish()`；未完成事件超过 8 MiB 时返回错误。`framing::eventstream` 提供 AWS 二进制帧解析并限制单帧大小。通常只在自定义传输/codec 集成时直接使用。

源码索引：[客户端](../src/client/mod.rs)、[共享请求/响应](../src/protocol/llm.rs)、[配置](../src/protocol/provider.rs)、[传输](../src/transport.rs)、[集成测试](../tests/)。可在本地生成逐项 Rust API 文档：

```sh
cargo doc --no-deps --open
```


详见[推理控制与服务档位价格](inference.md)：`info.features`、`info.pricing`、`price_quote`、`estimate_cost` 和 `estimate_stream_cost`。

## 宿主管理重试与账务

需要自行管理准入、取消和持久化账务的应用，可以调用
`prepare_on(profile, request, options, mode)`。profile 必须是具体连接名，不能是连接组。
准备阶段固定所选模型行、编码并认证请求，可能上传附件，但不会发送生成请求。
`PreparedCall` 不可克隆；宿主可先检查 `request()`、保存 `pricing_snapshot()`，
再通过 `dispatch_once()` 消耗该调用。每次重试或故障转移都需要新的 prepared call。

先检查 `ReceivedCall::status()`，再选择 `into_stream()` 或 `collect()`。
即使 HTTP 失败或回答内容损坏，收集结果的 usage、inference 仍可在 `decode()` 前读取。
调用 `finish()` 完成附件清理。`ModelStream::next_batch()` 按已收到的传输块返回观察值，
包括只有 usage 的块；解码后不会为了等待下一个块再次挂起。宿主应先保存这些观察值，
再让出执行权或验证应用自己的输出 schema。

`count_tokens_exact_in` 使用 Anthropic 精确计数接口。不支持的协议返回 `None`，
错误仍作为错误返回，计数不会生成回答。`FrozenPricing::estimate` 使用固定价格和已确认
的执行事实，保留币种，并拒绝不完整 usage 或未知 Fast 档位；它不负责应用账本。

Responses WebSocket 需要注入实现 `connect_websocket` 的 `Transport`。
通过 `PreparedCall::connect_websocket` 在准入前建立连接，再用
`dispatch_websocket_once` 发送一次 `response.create`。客户端不会自动重连或回退 HTTP。
`websocket` 模块提供续接状态辅助函数，连接生命周期属于宿主；已经尝试发送的调用必须
结算后才能发起回退调用。默认 HTTP transport 不提供 WebSocket 连接。

`CompletionRequest::controls` 提供采样、结构化输出、Anthropic context hint，以及
Responses 存储、续接和元数据选项。系统缓存控制保留 TTL 与 scope。原生内容携带协议
标识，不能跨协议重放。流式接口也提供原生注解 delta 和块结束事件；它们不是应用工具调用。
