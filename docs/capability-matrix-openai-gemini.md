# OpenAI / Gemini 能力证据矩阵

[English](capability-matrix-openai-gemini.en.md)

[机器可读矩阵](../data/capability-matrix-openai-gemini.json)记录截至 **2026-10-09** 的第一方文档证据，覆盖 OpenAI 和 Gemini 原始内置目录的全部模型，包括被普通 completion preset 过滤掉的 embedding、Realtime、图像、音乐、视频与 agent 条目。较早的模型与来源记录保留其原始核验日期。这是 P0 文档审计，不会更改运行时路由，也不代表客户端实现或真实账号已通过验证。

## 覆盖与含义

| Profile | 模型数 | 每模型操作数 | Supported | Unsupported | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openai` | 50 | 21 | 332 | 44 | 674 |
| `gemini` | 35 | 21 | 150 | 213 | 372 |
| 合计 | 85 | 1,785 个模型操作单元格 | 482 | 257 | 1,046 |

另保留 **3 个已移出目录的 Gemini 预览模型**于 `retired_models` 历史区，共 **63 个历史单元格**（9 Supported、12 Unsupported、42 Unknown），不计入现行覆盖。Flash-Lite Preview 于 2026-05-25 关停；Pro Image Preview 和 Flash Image Preview 于 2026-06-25 关停。[Google 退役记录](https://ai.google.dev/gemini-api/docs/deprecations)。

另有 **41 个独立服务操作**，覆盖向量库、缓存资源、File Search store、Gemini 模型目录、批任务、后台结果及音频服务；全部有正向文档证据。矩阵共引用 **97 个第一方来源**。服务存在不意味着每个目录模型都能调用它。

Gemini 原生 `models.list` 与 `models.get` 作为服务级操作独立记录，不增加模型能力列。官方列表返回 `supportedGenerationMethods`，客户端按 `embedContent` 筛选只是目录投影，不能推出所有返回模型都支持 embedding；每个模型的 embedding 能力仍由模型级文档证据决定。此 API 证据也不验证实时账户或区域可用性。

Gemini 批处理提交按 API 路由拆分为 `gemini.batch.generate_content.submit`（`batchGenerateContent`）和 `gemini.batch.embed_content.submit`（`asyncBatchEmbedContent`）。仅 `gemini-embedding-001` 和 `gemini-embedding-2` 在 embedding 专属操作中有正向证据；前者来自官方 Embeddings 文档对 Gemini Embeddings 模型族的 Batch API 说明，后者还在 Batch API 指南中有明确创建示例。其余模型的 embedding 批处理支持保持 `unknown`。创建是模型级操作；get、list、cancel、delete 和 results 是共享 Batch 资源的服务级操作。Embedding 批处理的 `interactions.batch` 是另一套 API，不能从 Batch API 支持推断。

- `supported`：官方文档明确支持该操作，或明确列出适用模型族／别名。
- `unsupported`：官方文档明确限制了该模型或端点；不会因文档没有提到功能就作此判断。
- `unknown`：本次查阅无法建立对应模型和操作的证据。来源指向查阅上下文，不构成“不支持”的证据。

所有行保持 `live_validation: not_run`、`account_region_validation: unknown`。范围是当前两个 profile 使用的第一方国际 API，不能外推到 Azure、Vertex、任意地域、订阅或账号。缺失的模型页、相近名称及 floating `latest` 别名都不自动继承其他模型的能力。

## 数据结构与使用

`operations` 定义操作、能力类别、模型／服务作用域和端点。`sources` 保存官方 URL、证据日期与定位章节。`profiles[].models[].cells` 为每个模型逐项列出三态、来源 ID 和判断依据；`services` 则记录独立资源操作；`retired_models` 保留已移除模型的原始操作证据和明确关停状态。每个 cell 的来源都必须能在注册表中解析。

`availability` 独立记录 `documented`、`restricted`、`deprecated`、`shut_down` 或 `unknown`。`documented` 仅表示官方页面列出了模型，不能视为当前账号可用。功能单元格描述文档中的模型契约；若模型已关停，`shut_down` 优先于任何历史功能支持，不能据此发起调用。

`audio.input` / `audio.output` 只描述模型音频模态；不会证明独立 STT、TTS 或 Live 端点存在。例如音乐输出不是语音合成，音频理解也不是文件转写接口。`responses.computer_use` / `generate_content.computer_use` 只说明工具契约；本地动作仍由宿主执行。

## 已锁定的差异

| 情况 | 矩阵处理 |
| --- | --- |
| `gpt-4o-2024-05-13` | JSON Schema 为 Unsupported；不会继承后来快照的 Structured Outputs。 |
| `gpt-5.2-pro`、`gpt-5.4-pro` | 模型页明确将 Structured Outputs 标为不支持；后者也不支持 Code Interpreter 和 hosted shell。 |
| GPT-5.6 及之后 | 显式缓存断点、`prompt_cache_options.ttl=30m` 单独列项；不会将较早的 `prompt_cache_retention=24h` 当作同一个控制。 |
| `gpt-5.6` | 官方明确指向 `gpt-5.6-sol`，才允许基于该别名关系记录能力。 |
| `gpt-5.3-codex-spark` | 查阅无法建立逐操作 HTTP API 证据，保持 Unknown。 |
| Gemini Interactions | 显式缓存与 Batch 明确不支持；独立缓存资源和 Batch API 的支持仍单独记录。 |
| Gemini 3 与 Deep Research | 普通 Gemini 3 的 Interactions Remote MCP 不支持；Deep Research agent 的 MCP 和必须使用后台执行分别有正向证据。 |
| 已退役 Gemini 预览模型 | Flash-Lite Preview（2026-05-25）、Pro Image Preview 和 Flash Image Preview（2026-06-25）移入 `retired_models`，不计现行目录覆盖。 |
| Gemini 2.5 Flash／Flash-Lite／Pro | 官方限制为曾活跃使用这些模型的用户；记录 `restricted`，没有检查账号资格。 |
| `lyria-3-pro-preview` | Interactions 清单明确列出此 ID，但尝试的模型页标识另一个模型，不复制该页功能表。 |

依据：[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[OpenAI 缓存](https://developers.openai.com/api/docs/guides/prompt-caching)、[GPT-5.4 Pro](https://developers.openai.com/api/docs/models/gpt-5.4-pro)、[Gemini Models API](https://ai.google.dev/api/models)、[Gemini Interactions](https://ai.google.dev/gemini-api/docs/interactions-overview)、[Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research)、[Flash-Lite Preview 关停公告](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview)、[Gemini 2.5 Flash 访问限制](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash)。其余具体来源见各 cell 的 `source_ids`。

OpenAI 后台模式允许 `store=false`，但会临时保存结果以便轮询；恢复流要求创建时已启用 `stream=true`。Gemini Interactions 的 `store=false` 与后台执行不兼容。两者不可共用同一条存储限制。[OpenAI Background](https://developers.openai.com/api/docs/guides/background)、[Gemini Interactions](https://ai.google.dev/gemini-api/docs/interactions-overview)。

目前 Gemini 模型卡中的通用“Caching supported”不足以确认每个模型接受显式 `cachedContent` 引用，因此这些模型级单元格仍可为 Unknown；`cachedContents` 的创建、读取、列举、更新与删除有独立 API 证据。补齐时应补模型和操作的证据，不用服务存在来填满模型矩阵。

## 维护与验证

增加原始目录模型时，补齐完整矩阵行；明确关停的模型移出目录时，将原始证据迁入 `retired_models`，保留关停来源。增加操作时，为该 profile 的每个模型添加 cell，缺证据先写 Unknown。不要把未运行真实调用改成通过；文档证据日期也不能作为 live 验证日期。

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 cargo test --test capability_matrix_openai_gemini
```

[测试](../tests/capability_matrix_openai_gemini.rs)检查原始 TOML 模型集合完全相等、逐模型操作齐全、七类缺口均有列项、三态依据合法、来源闭环与第一方域名、证据日期、独立服务作用域，以及快照／缓存代际／Gemini agent 与端点限制回归。测试不访问网络、不调用供应商，也不把可解析 URL 当作持续有效的在线链接。

OpenAI 的 `GET /v1/models` 独立记录为服务操作：返回账户可见模型与基础元数据，没有 Embedding 能力标记。客户端对已文档化 Embedding ID 的筛选不会改变其他模型的能力单元格，完整目录保留供调用方检查。[Models list](https://developers.openai.com/api/reference/resources/models/methods/list)。

GPT-Live 主 WebSocket 新增独立服务记录 `openai.live.connect`，与 Realtime 分开：固定 `/v1/live/sessions` 路由使用 session.start/started 和 session.close/closed，具有显式 Responses 委派及累计会话用量。服务证据不创建 Chat 模型行，也不代表账户可用性。[主连接指南](https://developers.openai.com/api/docs/guides/voice-websockets)。
