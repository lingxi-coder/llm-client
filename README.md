# llm-client

[English](README.en.md)

`llm-client` 是一个独立的 Rust 库，用于在应用中调用不同的 LLM 服务。应用使用统一的请求和响应类型，通过 provider 配置选择模型与连接；客户端负责协议编码、HTTP 传输、流式解析和错误分类。它也提供模型目录、托管式 Web Search、用量与费用估算等能力。

协议类型由本 crate 自身定义，可通过 `lingxi_llm_client::protocol` 使用。应用负责工具执行、权限、会话历史、凭证刷新和上下文压缩；接入示例见 [API 指南](docs/api.md#宿主工具执行与上下文恢复)。

## Features

- [推理控制与 fast 定价](docs/inference.md)：查询模型能力，设置 budget、effort 和 fast，读取标准／fast 价格及实际档位。Effort 影响用量，不改变单价。
- **统一多种模型协议**：内置 OpenAI Responses、Chat Completions、Anthropic Messages 和 Gemini 编解码器，以及 Azure、Bedrock、Vertex 等托管平台适配，减少应用内的协议适配代码。
- **通过配置接入服务**：内置 provider 与模型配置；兼容已有协议的服务可通过 profile 接入。同一 provider 可配置多个账号连接，分别同步模型并管理可见性。
- **路由与故障转移**：按模型解析连接，也可明确指定 profile；按需配置备用连接及其凭证，响应可用于追踪实际执行连接。
- **统一流式输出与搜索结果**：使用统一事件处理文本、工具调用和用量；通过 Web Search 接口获取服务端搜索结果及引用。搜索选项和模型能力按 provider 支持情况校验。
- **独立图像服务**：通过 `client.images()` 生成、编辑图片并查询原生异步任务；图像请求与模型能力和 Chat 分开。详见[图像生成指南](docs/images.md)。
- **用量与费用可追踪**：统一输入、输出和缓存 token 用量，支持根据模型目录价格及实际执行连接估算费用。
- **离线输入 token 估算**：按需启用供应商 tokenizer，估算 system、工具定义及会话文本等可见输入，并明确列出未计数的内容。
- **查询账户额度**：按连接读取供应商公开的余额、历史 Token 用量、额度窗口和编程套餐权益，区分 API Key 与登录用户，并标明数据的账户范围。
- **跨设备文件附件**：会话保存应用拥有的稳定附件引用，模型请求按当前连接自动解析为内联内容或受支持的 provider 文件引用；文件模型输入与跨设备预览保持独立。
- **开箱即用，也可扩展**：内置 HTTP 客户端与认证器，可替换传输、时钟或扩展协议。凭证由应用按请求传入，不写入本地 provider 配置。

### 支持的 LLM Providers

仓库在 [`data/providers/`](data/providers/) 中提供以下内置连接配置。表中的名称是调用 `complete_in()` 等方法时使用的 profile 名。

| 服务 | 内置 profile |
| --- | --- |
| OpenAI | `openai` |
| Anthropic | `anthropic` |
| Google Gemini | `gemini` |
| DeepSeek | `deepseek`、`deepseek-search` |
| Kimi | `kimi`、`kimi-intl`、`kimi-code`、`kimi-search`、`kimi-search-intl` |
| Qwen / Model Studio | `qwen`、`qwen-intl`、`qwen-us`、`qwen-hk`、`qwen-search`、`qwen-search-intl`、`qwen-search-us`、`qwen-search-hk` |
| MiniMax | `minimax`、`minimax-intl` |
| Z.AI / GLM | `zai`、`zai-coding`、`glm`、`glm-coding` |
| xAI Grok | `grok`、`grok-responses`、`grok-anthropic` |
| OpenRouter | `openrouter` |
| GitHub Copilot | `github-copilot` |

部分服务提供多个 profile，以适配不同协议、端点或账号类型。Azure、Bedrock 和 Vertex 提供协议适配器，可按需创建自定义 profile；它们不在上述内置连接列表中。具体模型及能力以 profile 配置、账号权限和服务端实际支持为准。

中国区与国际区的 Qwen、Kimi 和 MiniMax 连接使用独立 profile、API 地址和凭证变量，按实际开通的区域选择，不要跨区域复用 API Key。北京、新加坡、美国和香港的 Qwen Search profile 都支持 Responses API 的知识库搜索；`minimax` / `minimax-intl` 使用 Anthropic Messages 与 MiniMax 服务端 Web Search。

Remote 设备间共享图片、附件引用和 provider 文件生命周期的说明见[文件附件指南](docs/file-attachments.zh.md)。

构建 client 时必须通过 `with_region(Region::ChinaMainland)` 或 `with_region(Region::International)` 选择使用区域。provider/model 列表、模型解析和故障切换均按区域过滤；完整配置仍保留。自定义 profile 可通过 `regions` 声明可用区域，未声明时两区可用。详见[区域过滤](docs/api.md#region-区域过滤)。

默认构建不包含 tokenizer 后端及资产；按需启用方式见[离线估算输入 token](#6-离线估算输入-token)。公开扩展接口、用量报告和配置 v2 的变化见[架构与 API 迁移指南](docs/architecture-migration.md)。

## Getting Started

需要 Rust 1.94.0 或更高版本，以及可访问目标模型的 API key。下面使用内置 `openai` 连接完成第一次请求。

先了解三个概念：**provider** 是模型服务，**profile** 是具体连接配置（协议、地址、模型等），**client** 根据 profile 发送请求。应用负责提供凭证和会话消息；客户端负责协议转换、网络请求与响应解析。

### 1. 创建应用并添加依赖

```sh
cargo new llm-example
cd llm-example
```

在 `Cargo.toml` 的 `[dependencies]` 中添加以下依赖：

```toml
[dependencies]
lingxi-llm-client = { git = "https://github.com/lingxi-coder/llm-client", branch = "main" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
serde_json = "1"
```

### 2. 创建 Client 并发送请求

将以下完整示例保存为 `src/main.rs`：

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, Secret};
use lingxi_llm_client::{builtin_providers, LlmClientBuilder, RequestOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profiles = builtin_providers()?;
    let client = LlmClientBuilder::new(&profiles)?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
    let options = RequestOptions {
        credential: Some(Secret::new(std::env::var("OPENAI_API_KEY")?)),
        ..RequestOptions::default()
    };
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "gpt-4.1-mini",
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello! Introduce yourself briefly."}]
        }]
    }))?;

    let response = client.chat().complete_in("openai", &request, &options).await?;
    println!("{}", response.message.text());
    Ok(())
}
```

设置环境变量后运行应用，将得到模型返回的文本：

```sh
export OPENAI_API_KEY="your-api-key"
cargo run
```

这里由示例代码读取环境变量；客户端不会自动读取密钥。`openai` 是内置 profile 名，`gpt-4.1-mini` 是仓库内置目录中的模型 ID，需要你的账号有权访问。可通过 `client.providers()` 和 `client.chat().models()` 查看配置中的连接与模型。

对话通过 `client.chat()` 调用：`complete_in()` 指定连接，`complete()` 按模型自动路由，`stream_in()` / `stream()` 返回流式事件。`client.chat().models()` 会过滤元数据中声明图像输出的模型；`client.images().models()` 读取独立图像目录。原有顶层对话和流式方法仍可用，使用相同执行路径。详见 [ChatService](docs/api.md#chatservice) 和[流式响应](docs/api.md#流式响应)。

#### 流式 Chat 响应

如果需要流式输出，将 `main` 中的 `complete_in()` 调用及其后的 `println!` 替换为 `stream_chat(&client, &request, &options).await?`。添加以下函数并合并重复导入：

```rust,no_run
use std::io::{self, Write};
use lingxi_llm_client::protocol::{CompletionRequest, StreamEvent};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn stream_chat(
    client: &LlmClient,
    request: &CompletionRequest,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = client.chat().stream_in("openai", request, options).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
                io::stdout().flush()?;
            }
            StreamEvent::End { stop_reason, .. } => {
                eprintln!("\nStop: {stop_reason:?}");
            }
            _ => {}
        }
    }
    eprintln!("Profile: {}", stream.executed_profile());
    eprintln!("Usage: {:?}", stream.usage_report());
    eprintln!("Inference: {:?}", stream.inference_report());
    Ok(())
}
```

`stream_in()` 指定起始连接；`client.chat().stream(request, options)` 按模型自动路由。`ModelStream` 自带异步 `next()`，无需导入 `StreamExt`。打开流和读取事件都可能返回错误，示例通过 `?` 向上传递。收到 `End` 后仍继续读取，直到 `None`，再查看最终报告。

此示例只展示文本，忽略工具和思考事件。需要工具调用或重放 assistant 消息的应用，还应保留工具调用、原生内容和签名，见[流式响应](docs/api.md#流式响应)。用量可能缺失或不完整，需通过 `usage_is_complete()` 确认完整性。返回流句柄后，读取失败不会自动切换连接；丢弃流会释放底层响应。流式请求默认没有总时限，可用 `RequestOptions.total_timeout` 显式设置。

### 3. 配置、更新与读取模型

需要持久化配置时，将上例的 `client` 声明为 `mut`，并在构建后调用 `client.set_config_dir("./config")?`。该调用加载并保存目录中的 `providers.json`；应用重启时再次调用以恢复配置。

| 需求 | 接口 |
| --- | --- |
| 新增或替换连接 | `add_provider(profile)`；配置包含 profile 名、协议、地址和模型 |
| 读取连接配置 | `provider("openai")` / `profiles()` |
| 获取连接与可见模型列表 | `providers()` / `chat().models()` |
| 从服务端更新模型目录 | `sync_provider("openai", options.credential.as_ref()).await?` |
| 设置 provider 的模型白名单 | `set_tracked_models(provider_id, model_ids)` |
| 设置已跟踪模型的可见性 | `set_model_visibility(profile_name, model_id, visible)` |
| 删除连接或恢复内置配置 | `remove_provider(profile_name)` / `restore_builtin(profile_name)` |

每个账号连接分别同步，并提供该账号的凭证。完整配置示例与更新规则见[本地保存与多账号](docs/api.md#本地保存与多账号)和[模型目录](docs/api.md#模型目录)。

配置使用 v2；读取 v1 文件会返回 `UnsupportedVersion`，不会自动迁移。`add_provider()` 会完整替换连接；如需修改单个字段并让其他字段继续继承目录默认值，使用 `configured_models()` 获取行 ID，再调用 `set_model_override()` / `clear_model_override()`。详见[配置 v2 契约](docs/architecture-migration.md#配置-v2)。

### 4. 使用 Web Search

复用上例的 `client`、`request` 和 `options`，把消息改为需要联网查询的问题，在 `main` 中调用 `search(&client, &request, &options).await?`。将以下函数添加到 `src/main.rs`，合并重复的导入：

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, WebSearchConfig};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn search(
    client: &LlmClient,
    request: &CompletionRequest,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client
        .web_search_in("openai", request, WebSearchConfig::default(), options)
        .await?;
    println!("{}", response.message.text());
    if let Some(search) = response.web_search {
        for citation in search.citations {
            println!("{}", citation.url);
        }
    }
    Ok(())
}
```

`web_search*` 便捷方法属于 `LlmClient`。也可设置 `request.web_search = Some(WebSearchConfig::default())`，再将请求交给 `client.chat().complete_in()` 或 `client.chat().stream_in()`。

搜索由 provider 执行，需使用支持搜索的模型；启用搜索不保证每次请求都会触发搜索。域名限制、搜索引用和流式搜索的用法见 [Web Search 指南](docs/web-search.md)。

### 5. 生成图片

复用上例的客户端和凭据，在 `main` 中调用 `generate_image(&client, &options).await?`。添加以下函数并合并重复导入。图像服务使用独立的请求类型与模型目录：

```rust,no_run
use lingxi_llm_client::protocol::{ImageGenerationRequest, ImageRequestOptions};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn generate_image(
    client: &LlmClient,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = ImageGenerationRequest {
        model: "gpt-image-1.5".into(),
        prompt: "An orange cat on a blue background".into(),
        references: vec![],
        output: Default::default(),
        provider_options: Default::default(),
    };
    let image_options = ImageRequestOptions {
        credential: options.credential.clone(),
        ..Default::default()
    };
    let response = client.images().generate_in("openai", &request, &image_options).await?;
    println!("{} images", response.images.len());
    Ok(())
}
```

通过 `client.images().models()` 查看当前区域可见的图像模型。OpenAI、Gemini、Qwen、xAI、MiniMax、GLM/Z.AI 和 OpenRouter 已有内置图像路由；编辑、参考图、蒙版和原生任务能力依模型而定。

`submit_in()` 返回 `ImageTaskRef`，`get_task()` 每次查询一次；应用负责轮询和保存任务。结果中的临时 URL 需及时下载保存。图像请求不自动重试或切换连接，图像用量也不参与 Chat token 费用估算。详见[图像生成与编辑指南](docs/images.md)。

### 6. 离线估算输入 token

将上面的客户端依赖替换为以下配置，启用 OpenAI tokenizer：

```toml
[dependencies]
lingxi-llm-client = { git = "https://github.com/lingxi-coder/llm-client", branch = "main", features = ["tokenizer-openai"] }
```

```rust,no_run
use lingxi_llm_client::{LlmClient, LocalTokenCountError, protocol::CompletionRequest};

fn estimate_input(client: &LlmClient, request: &CompletionRequest) -> Result<(), LocalTokenCountError> {
    let estimate = client.estimate_local_tokens_in("openai", request)?;
    println!("{} input tokens via {}", estimate.input_tokens, estimate.tokenizer);
    if estimate.is_partial {
        println!("Not counted: {:?}", estimate.uncounted_components);
    }
    Ok(())
}
```

其他可选 feature 为 `tokenizer-deepseek`、`tokenizer-qwen`、`tokenizer-kimi`、`tokenizer-glm`；`tokenizers-all` 启用全部五种。仅支持已映射的模型：未启用对应后端时返回 `FeatureDisabled`，不支持的模型返回 `UnsupportedModel`。

估算不发送网络请求，也不读取附件。图片、远程文件和服务端隐藏状态等可能导致估算不完整，不能替代供应商返回的实际用量。覆盖范围与限制见[本地输入 Token 估算](docs/api.md#本地输入-token-估算)。

### 7. 错误排查

通过 `LlmError::kind()` 或错误变体分类处理，不要根据 `message` 文本决定控制流程。常见排查方向：

| 错误 | 排查方向 |
| --- | --- |
| `BuildError` | 检查重复 `profile_name`、协议是否有对应 codec、认证策略是否已注册，以及价格时间窗是否合法。 |
| `ModelUnavailable` / 路由歧义 | 核对模型 ID 是否存在于 profile；调用 `resolve()` 检查路由，或用 `complete_in()` / `stream_in()` 指定连接。 |
| `Authentication` / `PermissionDenied` | 确认 `RequestOptions.credential` 属于起始连接；故障转移凭证按备用 profile 名放入 `fallback_credentials`，并检查账户权限。 |
| `InvalidRequest` / `UnsupportedCapability` | 检查请求字段、provider 配置，以及所选模型是否支持工具、搜索或其他能力。 |
| `RateLimited` / `QuotaExceeded` | 检查 `retry_after` 和账户额度；故障转移默认关闭，客户端不对同一连接自动重试。 |
| `Transport` / `TransportTimeout` / `TlsCert` | 核对 endpoint、网络连通性、超时和 TLS 证书链。 |
| `StreamInterrupted` | 本轮流已中断；检查网络和 provider 状态。已有部分输出时不要盲目重放请求。 |
| `ProviderStoreError` | 检查配置目录权限、`providers.json` 格式，以及模型目录请求或写盘错误。 |

具体错误类型和处理建议见[错误处理](docs/api.md#错误处理)。Web Search 的未声明适配器、协议不匹配或不支持参数也会在请求发送前返回错误，见[搜索支持矩阵](docs/web-search.md#支持矩阵与参数)。

## Documents

| 文档 | 内容 |
| --- | --- |
| [API 接口指南](docs/api.md) | 客户端构建、请求、流式响应、认证、配置、模型目录与费用 |
| [推理控制与价格](docs/inference.md) | 能力查询、思考预算、effort、服务档位与实际费用估算 |
| [图像生成与编辑](docs/images.md) | 图像模型、生成、编辑、参考图与原生任务 |
| [架构与 API 迁移](docs/architecture-migration.md) | 扩展契约、用量报告、配置 v2 与 tokenizer features |
| [本地输入 Token 估算](docs/api.md#本地输入-token-估算) | 离线计数、支持模型与未计数内容 |
| [账户额度与用量](docs/api.md#账户额度与账户-token-用量) | 账户身份、查询预算、额度窗口与部分报告 |
| [文件附件指南](docs/file-attachments.zh.md) | 稳定附件引用、远程设备显示、resolver 与 provider 文件输入 |
| [Web Search 指南](docs/web-search.md) | 支持矩阵、搜索参数、引用、流事件与上下文重放 |
| [扩展接口](docs/api.md#扩展接口) | 接入新协议、模型目录与自定义认证 |
| [目录维护与发布](docs/maintenance.md) | 内置 provider / 模型目录更新及 crate 发布步骤 |
| [审查记录与已知边界](docs/review.md) | 已修复问题、设计决策与应用需要处理的边界 |

在本仓库运行 `cargo doc --no-deps --open` 可查看 Rust 类型与方法文档。

## Development

修改本项目时，从源码构建：

```sh
git clone https://github.com/lingxi-coder/llm-client.git
cd llm-client
cargo build --locked
```

内置 provider 配置位于 `data/providers/*.toml`。接入已有协议时添加 profile；新增协议时实现并注册 `WireCodec`，需要模型目录同步时再实现 `ModelDirectory`。详见[扩展接口](docs/api.md#扩展接口)。

提交改动前可运行：

```sh
cargo test --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked
cargo test --manifest-path tests/downstream-docs/Cargo.toml --locked --offline --target-dir target
```

CI 还会测试 `tokenizers-all` 配置，并分别编译每个 tokenizer feature。修改 token 估算时，额外运行 `cargo test --locked --features tokenizers-all`。

测试使用模拟传输和本地回环 HTTP 服务，不需要真实 API key。

## License

本项目采用 MIT 或 Apache License 2.0，任选其一。详见 [MIT License](LICENSE-MIT) 和 [Apache License 2.0](LICENSE-APACHE)。
