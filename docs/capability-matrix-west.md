# 西方厂商能力证据矩阵

[English](capability-matrix-west.en.md)

[机器可读矩阵](../data/capability-matrix-west.json)覆盖 OpenAI/Gemini 和中国厂商矩阵之外的所有内置目录：Anthropic、xAI、OpenRouter、GitHub Copilot，共 **4 个 provider、6 个 profile、408 个原始模型行、7,878 个模型操作单元格**。另有 **102 个独立服务记录、62 个第一方来源**。矩阵最新复核日期为 **2026-09-27**；旧记录保留各自证据日期。

| Profile | 模型行 | 每行操作 | 单元格 | Supported | Unsupported | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anthropic` | 12 | 20 | 240 | 121 | 25 | 94 |
| `github-copilot` | 33 | 14 | 462 | 0 | 0 | 462 |
| `grok-anthropic` | 7 | 18 | 126 | 3 | 4 | 119 |
| `grok-responses` | 7 | 14 | 98 | 26 | 11 | 61 |
| `grok` | 7 | 16 | 112 | 16 | 4 | 92 |
| `openrouter` | 342 | 20 | 6840 | 1246 | 658 | 4936 |
| 合计 | 408 | — | 7,878 | 1,412 | 702 | 5,764 |

这是文档证据快照，**不是运行时配置、实现完成清单或真实账户验收**。所有模型和服务均标记 `live_validation=not_run`、`account_region_validation=unknown`。公开 OpenRouter 目录的读取不调用模型，也不能验证账户权限、价格、区域、工具执行或实际响应。

## 读取方式

沿用 [OpenAI/Gemini 矩阵](capability-matrix-openai-gemini.md) 的 schema：`operations` 定义精确操作和 endpoint，`sources` 保存第一方 URL、核查日期及段落，`profiles[].models[].cells` 为每个模型逐操作记录证据，`services` 单列不属于 Chat 的服务。所有 profile 都覆盖结构化输出、缓存、服务端工具、检索、embedding、批量/后台及音频/实时七类；Anthropic 另有独立的 client toolset 操作。

- `supported`：来源明确支持该操作，或明确限定的模型族包含此行；不保证所有参数组合及上游路由都兼容。
- `unsupported`：来源明确排除该模型、endpoint 或参数组合。
- `unknown`：目前核查材料不足。缺失字段、目录中暂未出现、相似模型名称和页面抓取失败均不是不支持的证据。
- `availability` 单独记录。官方模型页或公开聚合目录中的 `documented` 只说明公开列出，不代表当前账户能够调用。

`batch.submit` 的模型单元格描述独立 Batch 服务对该模型的接纳范围，即使 profile 的主协议是 Messages，也不表示 `POST /messages` 能接受 Batch 参数。xAI 服务记录放在 `grok` 名下作为供应商服务索引，不向另外两个协议复制凭证或路由配置。

## 已确认的边界

Anthropic 的 JSON Schema、提示缓存和 Batch 对目录内模型有肯定证据。Haiku 4.5 支持代码执行，但不支持 programmatic tool calling；两者分开记录。Sonnet 5 未在所核查的 Tool Search 兼容表中明确出现，保留未知。Web Search、Web Fetch、MCP 不因其他工具支持就向全部模型扩散。JSON Schema 与 citations 的组合被明确排除。官方说明 Anthropic 没有自有 embedding 模型，因此不会把 Voyage 的能力记到 Claude 名下。[结构化输出](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[代码执行](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)、[Embedding](https://platform.claude.com/docs/en/build-with-claude/embeddings)。

稳定版 Browser 与 Computer client toolsets 是独立操作：`browser_toolset_20260801` 和 `computer_toolset_20260801` 的兼容表列出八个模型版本；对应 API IDs 为 `claude-fable-5`、`claude-fable-5-1`、`claude-mythos-5`、`claude-mythos-5-1`、`claude-opus-4-8`、`claude-opus-5`、`claude-opus-5-5`、`claude-sonnet-5`。其中 Fable 5、Fable 5.1、Opus 4.8、Opus 5、Opus 5.5、Sonnet 5 六行出现在当前原始目录；有限开放的 Mythos 5 和 Mythos 5.1 暂不在该目录，矩阵不伪造模型行。其余六个 Anthropic 目录模型仍为未知，不因相邻模型推断支持。平台级证据将 Google Cloud/Agent Platform 记录为支持，将 Microsoft Foundry 的这两个稳定版本记录为不支持；Foundry 仍可使用的旧版 beta computer tool 是另一项操作。以上是厂商文档契约，不代表客户端实现状态、账户权限或区域可用性。[Browser use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-use-tool)、[Computer use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)、[Claude on Google Cloud](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai)、[Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)。

xAI 三种主协议分别记录。Responses schema 的 `background` 被官方标为当前未使用的兼容字段；这里没有实测 `background=true` 是忽略还是拒绝。Deferred Chat 的结果只能读取一次，须在 24 小时内取回；这与持久化后台 Responses 不同。模型卡明确列出 Grok 4.20、4.20 Multi Agent、4.3 的 Batch 支持，并排除 4.5、4.6、4.7 和 Grok Build 0.1。Collections 管理使用 Management API key，搜索使用独立 inference API key。语音专用服务包含 REST TTS、双向 TTS WebSocket、内置音色列表，以及 Custom Voices 的创建、列表、读取、元数据更新、删除和参考音频下载。Custom Voices API 创建受 Enterprise 计划限制；官方页面注明 Custom Voices 仅在美国开放、伊利诺伊州除外。所有这些仍是服务级证据，实时账户及区域访问保持未知，也不据服务入口推断任何 Grok 目录模型支持语音操作。TTS 文档标注最后更新于 2026-09-19，Custom Voices 文档标注 2026-08-04。[TTS 与流式 TTS](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech)、[Custom Voices](https://docs.x.ai/developers/model-capabilities/audio/custom-voices)、[Responses reference](https://docs.x.ai/developers/rest-api-reference/inference/responses)、[Deferred](https://docs.x.ai/developers/advanced-api-usage/deferred-chat-completions)、[Grok 4.7 模型卡](https://docs.x.ai/developers/models/grok-4.7)、[Collections](https://docs.x.ai/developers/files/collections/api)。

OpenRouter 单列响应缓存、提示缓存和独立服务。Claude 的 `cache_control.ttl=1h` 与 GPT-5.6 及之后的 `prompt_cache_options.ttl=30m` 不混用。Chat 音频输入不等于独立 STT，Chat 音频输出不等于独立 TTS。Embedding、rerank、音频、Batch 均有独立服务证据；rerank 不表示托管向量库。Hosted shell 和 Tool Search 的文档限定 Responses/Messages，当前 Chat profile 对这些操作记录不支持。Batch 删除只处理终态资源，取消接口仍为未知。[缓存](https://openrouter.ai/docs/guides/best-practices/prompt-caching)、[服务端工具](https://openrouter.ai/docs/guides/features/server-tools)、[Batch](https://openrouter.ai/docs/batch-quickstart)。

GitHub Copilot 的 33 个原始模型行保留全部操作未知。公开 REST 文档说明管理和用量接口，CLI/SDK 产品文档说明 Agent 产品；这些不能证明内置 `api.githubcopilot.com` 原始 HTTP 路由具有底层 OpenAI、Claude 或 Gemini 的同名原生能力。[REST 文档](https://docs.github.com/en/rest/copilot)、[Agent 产品边界](https://docs.github.com/en/copilot/responsible-use/agents)。

Foundry 的服务证据区分托管方式和工具版本，不增加模型行，也不把平台支持复制到模型单元格。新增的五条记录如下：

| 服务记录 | 文档范围 |
| --- | --- |
| `anthropic.foundry.tool_search` | 两种托管方式均支持 regex/BM25 `20251119`；仍须满足工具的模型兼容表。 |
| `anthropic.foundry.azure.web_fetch` | Azure 托管仅支持 `web_fetch_20250910`；后续版本、动态过滤、`use_cache` 和 `response_inclusion` 被排除。 |
| `anthropic.foundry.anthropic.web_fetch` | Anthropic 托管支持当前四个版本：`20250910`、`20260209`、`20260309`、`20260318`，保留各自模型与参数限制。 |
| `anthropic.foundry.remote_mcp` | 两种托管方式均支持 `mcp-client-2025-11-20` beta connector；`mcp_toolset` 不是稳定版工具集。 |
| `anthropic.api.mcp_tool_list_pinning` | `mcp-client-2026-09-15` 的固定工具列表与 `mcp_tool_listing` 回放只在 Claude API 范围有肯定证据；Foundry 保持未建立，不声称实测拒绝。 |

Tool Search 和 MCP 的 [Features overview](https://platform.claude.com/docs/en/build-with-claude/overview) 条目没有仅限 Anthropic 托管的标记；[Web Fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool) 明确列出托管与版本差异；[MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector) 单独界定 2026 版功能范围。稳定 Browser/Computer toolset 的 Foundry 不支持记录保持不变；这些平台证据不代表真实部署或账户验收。

## OpenRouter 目录取证

核查使用 [所有输出模态的公开目录](https://openrouter.ai/api/v1/models?output_modalities=all)，避免默认只返回文本输出的过滤误差。342 个本地行中匹配 328 个，14 个缺失 ID 保存在 `scope.openrouter_directory.missing_catalog_models`，均保持未知，不据此标记退役。

仅肯定字段用于支持判断：`supported_parameters` 的 `structured_outputs` 支持 JSON Schema；仅有 `response_format` 不足以证明 JSON Schema。音频模态有 27 行输入、4 行输出证据；存在对应 `:batch` 变体的 63 个模型行取得 Batch 肯定证据。目录聚合的是模型能力；实际 endpoint 仍受 provider 选择及参数支持约束。动态路由器的元数据也不能保证固定上游模型。[模型目录说明](https://openrouter.ai/docs/guides/overview/models)、[结构化输出](https://openrouter.ai/docs/guides/features/structured-outputs)。

## 验证与维护

`tests/capability_matrix_west.rs` 检查剩余所有原始 provider TOML 行覆盖、七类通用操作与 Anthropic client toolsets、来源日期与第一方域名、来源/操作引用闭环、独立服务边界，以及上述关键限制。目录新增模型或 provider 时须补显式单元格，未知可以保留。

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 cargo test --test capability_matrix_west
```

测试仅验证本地证据结构和已记录事实的一致性，不在测试中联网、调用模型或验证账户。运行时支持程度应另查服务实现与对应契约测试。

2026-09-26 补充核查：客户端为第一方 Anthropic Code Execution 选用当前最新版本 `code_execution_20260521`，不需要 beta header；支持模型按官方兼容表核实。客户端的[类型化执行、文件输入及容器续用](anthropic-code-execution.md)单独实现，保留原生 usage，不将执行次数换算为时长费用；文档证据仍不代表真实账户验收。

Anthropic Skills 的创建、列举、查询、删除及对应版本操作已有独立服务证据；这不表示模型调用或账户验收通过。见 [Skills API](https://platform.claude.com/docs/en/api/http/skills)。

Foundry 另有四条按托管方式区分的 Code Execution / Programmatic Tool Calling 服务记录：Anthropic 托管明确支持，Azure 托管明确不支持。模型兼容、资源端点和账户权限仍须独立核验。[Foundry 托管限制](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)。

Files 服务操作另按 Foundry 两种托管方式逐项记录上传、列举、详情、下载、删除和 `ids[]` 批量 metadata 查询；Anthropic 托管支持，Azure 托管不支持。第一方 `ids[]` 查询也单列服务记录。下载只适用于可下载的生成文件，批量查询允许省略找不到的 ID；这些条件不证明账户可用性。[Files API](https://platform.claude.com/docs/en/build-with-claude/files)。

Foundry Skills 按托管类型分别记录：Anthropic 托管支持自定义 Skill 与版本的增删查和列表操作；Azure 托管不支持。版本内容下载在 Foundry 明确排除，Anthropic 托管也不例外。新增 18 条服务记录仍只表示文档证据，账户权限未实测。[Skills 概览](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)。


xAI 内置音色详情 `GET /v1/tts/voices/{voice_id}` 已单列操作证据，与自定义音色资源管理分开。真实调用未验收。
