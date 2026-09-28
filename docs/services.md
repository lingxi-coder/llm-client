# Chat 输出契约、缓存与 Embeddings

[English](services.en.md)

当前版本为 0.3.0。Chat 请求入口统一为 `client.chat()`，请求/响应类型为 `ChatRequest` / `ChatResponse`；没有旧类型别名或顶层 Chat 方法。持久化配置为 v3，旧版文件直接拒绝，且不改写原文件。

## 结构化输出

`OutputFormat::Text`、`JsonObject` 和 `JsonSchema` 区分普通文本、JSON 模式和 schema 约束。输出格式与函数工具的 `strict` 独立。OpenAI Chat、Responses、Messages、Gemini GenerateContent 及复用这些编码器的托管协议已实现 wire 映射；协议有编码能力不表示每个兼容服务、模型、区域或账户均支持该功能。

```rust
use lingxi_llm_client::protocol::{ChatRequest, ChatResponse, OutputFormat};
use serde_json::{json, Value};
let mut request: ChatRequest = serde_json::from_value(json!({
    "model": "your-model", "messages": [{"role":"user", "content":[
        {"type":"text", "text":"Return an answer object."}
    ]}]
}))?;
request.output_format = OutputFormat::JsonSchema {
    name: "answer".into(), strict: true,
    schema: json!({"type":"object", "properties":{"answer":{"type":"string"}},
        "required":["answer"], "additionalProperties":false}),
};
fn parse(response: &ChatResponse, request: &ChatRequest)
    -> Result<Value, lingxi_llm_client::protocol::StructuredOutputError>
{
    response.structured_json(&request.output_format)
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`structured_json()` 校验 JSON 和 schema；`structured::<T>()` 在校验后反序列化为宿主类型。拒答、截断、工具调用和暂停状态不能作为最终结构化结果。错误保留完整 `ChatResponse`，包括文本、用量和来源。流式请求可直接调用 `stream.collect_structured_json(&request.output_format)` 或 `stream.collect_structured::<T>(&request.output_format)`；该接口消费到终止事件后才校验，返回重建的文本响应和原始事件列表。失败时 `StructuredStreamError` 保留已收到事件，校验失败还保留响应。调用方应传入与请求相同的输出契约；需要逐事件处理时仍调用 `stream.next()`。

schema 不做自动改写或约束删除。已知不支持的子集和原生配置冲突会报错；服务端仍负责模型特有的限制。JSON Schema 校验器禁用网络和文件解析，未知格式名报错。`extra.body` 不能注入或扩展输出契约；非冲突兄弟字段可保留。局部 token 估算计入 schema，并标注无法估算的服务端格式指令。

Claude Messages 的严格工具与 JSON 输出共用服务端 schema 编译预算。发送前校验整个请求：最多 20 个 `strict` 工具、所有严格工具和 JSON 输出合计最多 24 个可选参数、16 个 `anyOf` 或类型数组参数；普通非严格工具不计入。客户端不会删改 schema 来绕过限制。[Claude 结构化输出限制](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)。

Claude 的严格工具输入与严格 JSON 输出都会在发送前拒绝递归本地 `$ref`、直接作为 `allOf` 分支的 `$ref`，以及官方列表之外的字符串 `format`（`date-time`、`time`、`date`、`duration`、`email`、`hostname`、`uri`、`ipv4`、`ipv6`、`uuid`）。严格 JSON 输出不能与输入中的原生 `document` / `search_result` 块的 `citations.enabled: true` 或最后一条 assistant 预填消息组合；这些组合会被拒绝。普通文本里提到类似 JSON 的内容不算启用引用。[Schema 限制](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[引用功能](https://platform.claude.com/docs/en/build-with-claude/citations)、[消息预填](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)。

OpenAI 的 `gpt-4o-2024-05-13` 可以使用 JSON Object 模式，但其快照早于官方支持 JSON Schema 输出的版本；该组合会在发送前拒绝。[OpenAI 结构化输出说明](https://developers.openai.com/api/docs/guides/structured-outputs)。

在一方 DashScope OpenAI 兼容 Chat 路由上，Qwen 的 JSON Object 模式要求 system 或 user 的显式文本中包含大小写不敏感的 `JSON`。只有 `req.system` 和 system/user 角色的 `ContentBlock::Text` 会满足此条件；assistant 历史、工具描述和文档/附件内容不计入。当前客户端目录已确认的严格 JSON Schema 模型为 `qwen3.8-flash` 和 `qwen3.8-max`。这些模型会保留可选属性及 schema 中的 `additionalProperties` 设置，同时继续校验其余已支持的 schema 子集。此策略不扩展到通用 OpenAI 兼容网关或 Qwen Responses。[Qwen 结构化输出](https://help.aliyun.com/en/model-studio/qwen-structured-output)。

一方 DeepSeek OpenAI 兼容 Chat 路由上的 JSON Object 模式同样要求 system 或 user prompt 包含 `json`。发送前只检查 `req.system` 和 system/user 消息中的显式 `ContentBlock::Text`，大小写不敏感；assistant 历史、工具描述及文档/附件不计入。此规则只应用于官方 `api.deepseek.com` HTTPS 路由。DeepSeek 文档还建议提供目标 JSON 示例并合理设置 `max_tokens`；客户端不会生成或猜测这些内容。[DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)。

对应官方 wire 文档：[OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs)、[Claude](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[Gemini](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)、[Kimi JSON Mode](https://www.kimi.com/help/kimi-api/api-model-capabilities)、[GLM 结构化输出](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output)。

## 提示缓存

`SystemBlock` 仅保存文本。`ChatRequest.prompt_cache` 集中保存自动缓存 TTL 与断点，断点索引引用原始请求中的工具、system 或消息内容块。空策略保留服务端默认行为，不代表禁用服务端自动缓存。

```rust
use lingxi_llm_client::protocol::{CacheBreakpoint, CachePosition, CacheTtl, PromptCachePolicy};
let policy = PromptCachePolicy {
    automatic: Some(CacheTtl::FiveMinutes),
    breakpoints: vec![CacheBreakpoint {
        position: CachePosition::System { index: 0 },
        ttl: CacheTtl::OneHour,
    }],
    ..PromptCachePolicy::default()
};
```

OpenAI Responses 使用独立的原生缓存字段：GPT-5.6+ options 管理最短生命周期，文档列出的模型可另设独立最大 retention；详见 [OpenAI Responses 提示缓存](openai-responses-prompt-cache.md)。其他显式策略编码支持 Messages 系列协议、Qwen Chat 和按模型限制的 OpenRouter Chat 适配：OpenRouter Anthropic 路由使用五分钟或一小时 TTL；列出的 Alibaba/Qwen OpenRouter 模型只接受五分钟内容断点；OpenAI GPT-5.6 及后续模型只接受三十分钟 TTL。`CacheTtl::ThirtyMinutes` 在其他路由会于发送前拒绝。MiniMax 限定为五分钟显式断点；其他协议拒绝显式策略。校验覆盖断点范围、重复位置、数量、可缓存文本块和长 TTL 顺序。不会伪造命中、自动重建远端缓存或推算缓存折扣；使用提供方实际返回的用量。[Claude 缓存协议](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)、[MiniMax 缓存协议](https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache)、[OpenRouter 提示缓存](openrouter-prompt-cache.md)。

独立的 [Gemini Context Cache 服务](gemini-context-cache.md) 已支持 Developer API 显式缓存的创建、查询、列举、更新与删除；不据此推定其他提供方支持相同生命周期。

OpenRouter 的**网关响应缓存**与上述 prompt 缓存分开：在 Chat 的 `RequestOptions.openrouter_response_cache` 设置 `OpenRouterResponseCache::Enabled { ttl_seconds: Some(600), refresh: false }`，即可发送官方缓存头；该策略适用于配置为 OpenRouter 官方 Chat Completions、Responses 或 Anthropic Messages 协议的请求。`Disabled` 会显式关闭。TTL 限定 1–86400 秒，刷新仅替换当前请求的缓存项。远端 preset 显式关闭缓存时，启用请求头不能覆盖它。该策略只允许 OpenRouter 官方 HTTPS 路由，并固定本次请求的连接，防止故障切换改变缓存键。客户端会把官方 `X-OpenRouter-Cache-Status: HIT|MISS` 响应头投影到 `ChatResponse.response_cache`；命中时还可包含缓存 age、剩余 TTL 与来源 generation ID。流式结果可用 `ModelStream.response_cache()` 读取相同投影。缺少或无法识别状态头时为 `None`；不会从零 usage 推断命中。[OpenRouter 响应缓存](https://openrouter.ai/docs/guides/features/response-caching)。

OpenRouter Embeddings 的官方 `https://openrouter.ai/api/v1/embeddings` 路由也接受同一请求级策略，结果位于 `EmbeddingResponse.response_cache`。非 OpenRouter 或非官方 Embeddings 路由会在发送前拒绝该策略；缺少或未知缓存状态不推断为命中。

## 文本 Embeddings

`client.embeddings().embed(profile, request, options)` 使用 `ProviderProfile.embeddings` 指定的完整操作 URL 和鉴权，不从 Chat URL 推导。服务配置支持 `ServiceSetting::Inherit`、`Enabled(route)`、`Disabled`；配置重新加载会继承当前定义，显式禁用不被目录更新覆盖。

```rust
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::embeddings::{EmbeddingRequest, EmbeddingResponse, EmbeddingError};
async fn embed(client: &LlmClient, options: &RequestOptions)
    -> Result<EmbeddingResponse, EmbeddingError>
{
    client.embeddings().embed("openai", &EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: vec!["A document to index".into()],
        dimensions: Some(256), task: None,
    }, options).await
}
```

实现 OpenAI、OpenRouter、Gemini batchEmbedContents、Qwen DashScope 四种文本编码，内置 OpenAI、OpenRouter、Gemini 和 GLM 路由。OpenRouter 将 `RetrievalQuery` / `RetrievalDocument` 分别映射为 `input_type: search_query` / `search_document`，并要求浮点向量；其余 task 在发送前拒绝。Qwen 需要显式配置当前工作空间对应的完整 endpoint。Embedding 模型名直接用于该服务，不通过 Chat 模型目录路由。

OpenRouter 内置 profile 另配置了独立 `models_endpoint`。调用 `client.provider::<OpenRouterClient>("openrouter")?.embeddings().list_models(offset, limit, options)` 可按官方 `offset`/`limit` 分页，`limit` 为 1–1000；返回 `EmbeddingModelPage`，包含 ID、名称、上下文长度、输入模态、原生元数据，以及服务端提供时的 `total_count` 和 `next_offset`。目录只用于发现，不会自动改写 Chat 目录或推断某模型支持当前请求参数。

Gemini 也有独立的 `models_endpoint`，通过 `GoogleClient::embeddings().list_models` 和 `get_model` 发现明确声明 `embedContent` 的模型，保留精确资源 ID、基础模型 ID 及原始元数据。分页使用绑定原请求范围的原生 token；过滤后为空的页面仍可能有下一页，客户端不会自动翻页。用法与账户范围边界见 [Gemini 模型目录](gemini-embedding.md#发现嵌入模型)。其他提供方的独立 Embedding 模型目录仍待核实。

返回向量按输入索引排序；缺失/重复/越界索引、非有限数字、空向量和维度不一致均报错。不会自行归一化、截断、拆批或重试。服务端未返回模型、用量或价格时不会猜测；用量保留原始字段并区分完整、部分、缺失和无效。请求和响应最多 64 MiB，统一 deadline 覆盖自定义 transport。通用 `EmbeddingRequest` 输入为文本；Gemini Embedding 2 的多模态输入使用独立的 Gemini 专用接口，其他提供方的支持不从此推断，详见 [Gemini Embeddings](gemini-embedding.md)。

官方参考：[OpenAI](https://developers.openai.com/api/reference/resources/embeddings/methods/create)、[Gemini](https://ai.google.dev/api/embeddings)、[Qwen](https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api)、[OpenRouter Embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings) 与 [模型目录](https://openrouter.ai/docs/api/api-reference/embeddings/list-all-embeddings-models)。

## 原生输出数据

OpenRouter Responses / Messages 的正则工具搜索与托管 Shell 已有独立类型化配置，包含延迟加载、容器引用作用域和禁止自动重试的执行规则，见 [OpenRouter 服务端工具](openrouter-server-tools.md)。Chat Completions 不支持这两个入口。

`ChatRequest.tools` 声明由宿主执行的函数；`ChatRequest.hosted_tools` 声明由提供方执行的工具。除 Web Search、File Search 和 OpenAI Code Interpreter 外，Gemini GenerateContent 还提供 `GoogleHostedTool::CodeExecution`、`GoogleHostedTool::UrlContext` 和 `GoogleHostedTool::MapsGrounding(GeminiMapsGroundingConfig { enable_widget, lat_lng })`。Gemini 选项只对 Google Gemini Developer API 的精确 `GeminiGenerateContent` 路由开放；Vertex、网关、其他自定义端点和未知模型会在附件准备与凭据使用前拒绝。支持的模型为 `gemini-2.5-flash` / `flash-lite` / `pro`、`gemini-3-flash-preview`、`gemini-3.1-flash-lite` / `pro-preview`、`gemini-3.5-flash` / `flash-lite`、`gemini-3.6-flash`、`gemini-3.7-flash` 和 `gemini-3.8-flash`。客户端函数只可与这些内建工具在 Gemini 3 模型上组合，并会启用官方要求的服务端工具上下文传递及 `VALIDATED` 函数调用模式；Gemini 2.5 组合会预先拒绝。Maps 可选 `lat_lng` 和 `enable_widget`，且拒绝多模态输入；Maps 与 Google Search 的组合只开放给 Gemini 3.5 Flash 及后续列出的 Flash 模型。多个新 Gemini 内建工具不会被自动组合。为避免重复执行服务端工具，含这些 Gemini 工具的请求固定到选中的连接；遇到网络错误、服务端错误或文件缺失时，客户端不会自动重发。重复托管工具会在附件解析或网络调用前报错。

OpenAI Responses profile 支持 `OpenAiHostedTool::CodeInterpreter(CodeInterpreterConfig { memory_limit: Some(CodeInterpreterMemoryLimit::FourG), ..Default::default() })`。未设置 `container` 时使用自动容器；省略内存档位时使用服务端默认值，也可选择 1g、4g、16g 或 64g。`CodeInterpreterConfig::with_files()` 可将已有、具备账户作用域的 OpenAI Files API 引用作为 `container.file_ids` 挂载；请求需提供创建这些文件引用时使用的同一稳定 `RequestOptions.file_account_scope`。客户端会在发送前检查连接身份、就绪状态和过期时间。若要复用已有容器，可将 `OpenAiContainerRef` 传给 `CodeInterpreterConfig::with_container()`，并提供创建该引用时使用的同一稳定 `RequestOptions.account_scope`。显式复用仅支持官方 OpenAI Responses endpoint，wire 只发送容器 ID；不能同时设置仅适用于自动容器的 `memory_limit` 或 `files`。显式容器所需文件应在 Responses 请求前通过[Containers 服务](openai-containers.md)附加。该服务也可在创建显式容器时设置 `expires_after`；Responses 自动容器配置没有过期选项，过期容器会被提供方拒绝。请求自动包含 `code_interpreter_call.outputs`，返回的调用项保留为 `ProviderContent`。Code Interpreter 请求会固定到选中连接，发送后遇到不确定结果时不会自动重放或故障转移。token 价格估算不包含工具及容器费用；容器和模型支持范围仍由提供方校验。[OpenAI Code Interpreter 官方说明](https://developers.openai.com/api/docs/guides/tools-code-interpreter)。

Responses 的托管工具及未知输出项保留为 `ProviderContent`，流式 `output_item.done` 与最终响应重复出现的同一输出项只发出一次。Gemini 返回的 `executableCode`、`codeExecutionResult`、`inlineData`、`toolCall` 和 `toolResponse` Part 也保留原生 JSON，可在同一 Gemini 协议上原样续传；它们不会转换成宿主执行的函数调用。URL Context 与 Maps 的候选级元数据位于 `ChatResponse.web_search.metadata`，不会伪装成内容 Part；Maps 来源保留 URI、标题、place ID 和归属数据，渲染时仍须遵守 Google Maps 官方归属要求。MCP 审批请求标记为 `StopReason::Other("requires_action")`；宿主负责批准与后续处理，库不会执行这些内容。各 provider 的原生工具通过 `.into()` 转换为 `HostedTool::Native`；各 codec 仅编码其明确支持的格式。

## Responses 续传

为续传请求设置 `RequestOptions.account_scope`，使用同一账户的稳定、非密钥标识。成功的 `ChatResponse.continuation` 返回 `ContinuationRef`；流式请求需读到终止事件后调用 `ModelStream::continuation()`。下一次请求将该引用赋给 `ChatRequest.continuation`，只在 `messages` 中加入新输入，并继续使用相同 `account_scope`。如果前一请求走了备用连接，请使用 `response.executed_profile` 指定该连接。

引用绑定 provider、profile、端点、wire 模型、账户及 Qwen 检索工作空间。绑定不符会在附件解析和 HTTP 前报错；续传不会故障转移，也不会在文件 404 后自动重提。未设置 `account_scope` 时仍能读取原始 `response_id`，但不会得到可复用的引用。直接通过 codec 解码的响应没有客户端路由信息，因此也不会生成引用。此接口不表示所有 Responses 兼容服务都持久化响应；profile 仍须声明 `extra.supports_previous_response_id = true`。

整体规划的实际完成状态见 [实施清单](implementation-plan.md)。所有新增 API 均尚未通过真实账户验收。
