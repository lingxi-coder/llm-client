# 架构与 API 更新

[English](architecture-migration.en.md)

0.3.0 保留单个独立 Rust crate，采用按 provider 组织的 client 和资源模块。旧入口和模块路径直接删除，不提供别名或转发兼容层。持久化配置仍为 v3；不兼容旧配置和旧响应格式。读取 v1/v2 会返回 `UnsupportedVersion`，不会改写原文件。

## Provider Client 与共享执行

公开调用链为 **统一服务 → Provider Client → 共享 codec 与 runtime**。`providers/<provider_id>` 拥有原生资源、类型、能力校验及 wire 差异；`codecs` 实现共享协议，`hosting` 放置 Azure、Bedrock、Vertex 和 Foundry 适配；`runtime`、`transport`、`auth` 共享快照、deadline、连接和请求认证。

Chat、Images、Embeddings 保留统一入口。原生 Audio、Batch、Background、Vector Stores、Interactions、File Search 和知识库通过对应的 typed client 访问，资源方法不再重复接收 profile。`provider::<T>(profile)` 只接受确切的 profile，拒绝 provider 身份不匹配和路由组，也不要求该连接存在 Chat 模型行。provider 身份与区域、Search、Coding 和协议变体配置相互独立。

Typed Chat 的故障切换只使用相同 provider ID 的备用 profile；统一 Chat 仍按显式路由组策略选择备用连接。缓冲文件上传在发送后遇到传输失败、5xx 或无法解码的成功响应时返回 `FileUploadOutcomeUnknown`，避免将结果不明的上传用于自动故障切换；Gemini 上传及恢复轮询在已取得文件 ID 后超时，返回携带可恢复引用的 `ProviderFileProcessing`。绑定的 Anthropic 与 Google Batch 服务使用该 profile 注册的鉴权器。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::OpenAiClient;
use lingxi_llm_client::providers::openai::audio::{AudioInput, TranscriptionModel, TranscriptionRequest};

async fn transcribe(client: &LlmClient, input: AudioInput, options: &RequestOptions)
    -> Result<(), Box<dyn std::error::Error>>
{
    let provider = client.provider::<OpenAiClient>("openai")?;
    let request = TranscriptionRequest::new(TranscriptionModel::GptTranscribe);
    let result = provider.audio().transcribe(input, &request, options).await?;
    let _ = result;
    Ok(())
}
```

Live provider 句柄在每次操作开始时捕获最新快照；从 `ClientSnapshot` 获得的句柄始终使用固定快照。原生引用保留明确的账号、workspace、区域和 endpoint scope。构造器不捕获密钥，每次调用通过 `RequestOptions` 传入凭证。原生路由来自显式 scope 或资源配置，不从 Chat URL 猜测。

`protocol::NativeExtension` 保存格式标识和无损原生 JSON。provider 自有类型负责构造和校验类型化视图；原生托管工具转换为 `HostedTool::Native`。签名、工具身份和未知回放字段保持完整，provider 专属资源类型不再位于 `protocol`。

共享执行保留 `prepare → seal → dispatch_once`、冻结价格、错误后的用量结算、取消和附件清理。Provider 不另建连接池或自动重试生成请求；故障切换由统一层负责，并要求每个备用 profile 单独提供凭证。参考实现记录见[固定版本的官方 SDK 参考](provider-sdk-references.md)。

## 服务与执行

`LlmClient` 是可低成本 clone 的请求句柄，共享运行资源和配置发布槽。`build_managed()` 独立返回 `ClientConfigManager`，其异步方法串行化管理操作，不锁住请求。Live 操作各自捕获不可变状态；`client.snapshot()` 可把全部服务和查询固定到同一版本。发布同时安装配置、账户绑定及缓存代际，在途请求和旧 snapshot 保持原状态。repository 负责文件锁、重读、格式验证和原子替换。详见[共享 Client 与迁移](client-reuse.md)。

完整响应和流式响应共用路由、附件解析、准备及认证。普通请求可按策略文件失效重试或 failover；continuation 固定原连接且不会自动重提，备用连接使用独立凭证。单调时钟 deadline 覆盖异步认证、上传、轮询、读取和清理。完整响应保留默认 120 秒（视频保留原有较长策略），流式响应默认无总时限。墙上时钟只用于时间戳与价格。

## 扩展接口变化

| 接口 | 契约 |
| --- | --- |
| `Transport` | 只实现异步 `send(HttpRequest) -> Result<StreamResponse, LlmError>`，返回原始字节和全部 HTTP 状态；禁止自动重定向与自动重试。 |
| `HttpExecutor` | 公共响应收集、大小限制和 deadline；错误保留状态、headers 和最多 64 KiB 正文。 |
| `WireCodec` | `encode_request(EncodeRequest, &CodecContext)`、`encoded_body_len`、`decode_response`、`stream_decoder`。默认计长采用实际编码；内置 codec 共用序列化与计数 writer。 |
| `EncodeRequest` | 借用原请求、共享媒体和逐块绑定。自定义 codec 编码每个块前调用 `block(original)`，通过 `inline_media` 取得应用附件字节。 |
| `CodecContext` | 连接数据（含连接能力限制）、wire model、`RequestMode` 和文件账户 scope；不携带凭证值、路由链或文件服务。同一 wire ID 对应多行时，使用 `CodecContext::for_model` 指定已选中的模型行。 |
| `StreamDecoder` | `push_bytes`、`finish`、`usage_report`，接受任意网络分片；返回有序 `Vec<Result<StreamEvent, LlmError>>`，保留同一分片中错误之前的有效事件。终止或错误后忽略后续输入。 |
| `RequestOptions` | 删除 `stream`；模式由调用入口或 codec context 决定。 |
| `UsageReport` | `Option<Usage>` 与 `Missing / Partial / Complete / Invalid`，完整响应和流共用。 |
| `AccountUsageSource` | `fetch(&AccountFetchContext, &mut AccountReport)`，取得指标就写入；超时只使尚未完成字段失败。 |

删除重复的 profile-aware codec 接口、`FrameStream`、`UrlOpener` 和 Transport 未使用的 WebSocket 接口。普通 decoder 持有 SSE 分帧器，Bedrock 持有 EventStream 分帧器。Hosted adapter 在基础请求最终序列化前修改字段。native reasoning envelope、签名、工具 ID、多模态顺序及消息／附件引用结构保持不变。

计价时保留执行请求所用的 snapshot；live client 的价格查询使用查询时配置。实际成本从响应中的完整报告读取：

```rust
use lingxi_llm_client::{ClientSnapshot, ResolvedRoute};
use lingxi_llm_client::protocol::{ChatResponse, LlmError, Submission};
fn report(snapshot: &ClientSnapshot, route: &ResolvedRoute, response: &ChatResponse)
    -> Result<(), LlmError>
{
    let cost = snapshot.estimate_actual_cost(route, response, Submission::Interactive)?;
    println!("{cost:?}");
    Ok(())
}
```

流结束后调用 `estimate_stream_cost(&route, &stream, submission)`；部分或无效报告返回 `CostUnavailable`。请求前的 `estimate_cost` 仍可接受原始 `Usage` 作为估算输入。

## 配置 v3

静态定义、用户配置、账户观测与运行快照各司其职。builder 的 profile 切片提供 caller 默认值；`add_builtin_profile(name)` / `add_builtin_profiles()` 显式注册 builtin 定义，`restore_builtin(name)` 也使用 builtin 定义引用。引用明确记录 `builtin` 或 `caller`，不会比较值来猜测来源。

描述、上下文和输出上限按 **用户覆盖 > 账户观测 > 当前定义** 合并；其他字段按 **用户覆盖 > 当前定义** 合并。`add_provider` 是完整替换，模型字段均视为显式设置。需要跟随目录时使用字段覆盖。`Clear` 设置字段的空值／默认值；`Inherit` 移除覆盖，保留有效观测。

```rust
use lingxi_llm_client::{ClientConfigManager, ProviderStoreError};
use lingxi_llm_client::configuration::{FieldOverride, ModelField};
async fn edit(config: &ClientConfigManager, profile: &str) -> Result<(), ProviderStoreError> {
    let rows = config.configured_models(profile).await?;
    if let Some(row) = rows.first() {
        config.set_model_override(profile, &row.row_id, ModelField::Description,
            FieldOverride::Set(serde_json::json!("My model"))).await?;
        config.clear_model_override(profile, &row.row_id, ModelField::Description).await?;
    }
    Ok(())
}
```

条目保留顺序和独立行标识，相同 wire ID 不合并。定义中同时具有相同 wire ID 和显示名称的重复行按出现序号识别；更新这些完全同名条目时，调用者应保持相对顺序。按 wire 设置可见性遇到多行会拒绝，按 row ID 的修改不会歧义。`replace_model` 替换原行并抑制旧目录条目复活；已观测模型后来进入静态目录时接入默认值，不生成重复行。

allowlist 控制跟踪、持久化和展示，不改变显式路由规则。兼容性单独控制执行，仍跟踪的模型元数据不会因暂时不兼容而丢失。备用定义快照按 allowlist 过滤，并在成功提交后刷新；只有整个定义缺失时才使用，包括空 builder 恢复，不能借它复活当前目录已移除的模型。

事务加锁后重读最新状态，验证候选，同步临时文件并原子替换。写入失败不安装新运行快照，不保存凭证值。目录同步保留分页保护、跨区域账户管理与 `ProfileChanged` 检查。已删除 v1 迁移、legacy baseline 和备份逻辑。

## 账户与 tokenizer features

`AccountQuery.execution` 默认总预算 60 秒，单次 HTTP/RPC 30 秒。预算从取得执行槽位开始，分页和后续调用共用。批量默认并发 4，通过 `with_account_concurrency(NonZeroUsize)` 调整。查询无序完成，返回结果保持 profile 原顺序。

```toml
[dependencies]
lingxi-llm-client = { version = "0.3", features = ["tokenizer-openai", "tokenizer-qwen"] }
```

可选 feature：`tokenizer-openai`、`tokenizer-deepseek`、`tokenizer-qwen`、`tokenizer-kimi`、`tokenizer-glm`，以及 `tokenizers-all`；`default = []`。已知模型未启用 feature 返回 `FeatureDisabled`，未知模型返回 `UnsupportedModel`。源码包保留资产与许可证，默认构建不包含对应后端和资产。验证结果与测量见[架构验收记录](architecture-validation.md)。
