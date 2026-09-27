# 中国厂商能力证据矩阵

[English](capability-matrix-china.en.md)

[机器可读矩阵](../data/capability-matrix-china.json) 是 2026-09-27 的第一方文档证据快照，不参与运行时路由，不代表实现完成或账户验收。所有记录均为 `live_validation: "not_run"`，账户与区域权限仍为 `unknown`。

## 覆盖与读取

矩阵覆盖 Qwen、DeepSeek、Kimi、MiniMax、GLM/Z.AI 的全部 21 个内置 profile、66 个原始 `[[model]]` 行和 1,254 个模型操作单元格。另有 606 个独立服务记录，其中 24 个记录保留 12 个独立图像模型行的生成/编辑维度。覆盖完整不等于每个单元格都有肯定结论。

- `operations` 定义操作、所属能力、`model`/`service` 范围；`sources` 保存第一方 URL、核实日期和具体章节。
- `profiles` 保留原始目录路径、协议、URL、地区和账户产品。Qwen 北京、新加坡、香港、美国，GLM 大陆、Z.AI 国际以及 Coding Plan 分开。
- 每个模型的 `cells` 包含 `supported`、`unsupported` 或 `unknown`，并引用 `source_ids` 和证据依据。日期与线上验证状态继承自模型记录。
- `availability` 独立记录公开在售文档、未知或 `shut_down`。已从运行目录移除的 7 个 Kimi 模型记录（大陆 6 个、国际 1 个）保存在顶层 `retired_models` 历史区，不计当前覆盖。历史能力不覆盖已退役状态；`documented` 也不保证某个账户当前可调用。
- `services` 使用独立操作身份。Embedding、知识库、批处理、ASR、TTS、音色管理、异步音频和 Realtime 的成功证据不会赋给同 profile 下的 Chat 模型。`documented_models` 是已核实的服务模型示例或明确允许列表，不是自动生成的完整目录。某项服务操作可通过 `profile_ids` 限定到有证据覆盖的 profile。

`unsupported` 仅用于第一方明确排除、拒绝或忽略的功能，以及已退役模型的创建操作。未列出、抓取失败、模型名称类似和协议兼容都不能证明不支持。未知状态保留后续核实工作；无需用兼容层或旧版 API 填补。

## 会影响实现的差异

| 场景 | 证据结论 |
| --- | --- |
| Qwen Responses | 支持多个原生工具；`background` 明确不支持，Batch 是独立 API。区域与 workspace 路由需单独选择。[Responses](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-responses) |
| Qwen Batch / RAG | 北京与新加坡的 Batch 模型列表不同；北京支持的 Qwen 3.8 不能自动扩展到新加坡。RAG REST 已核实北京 workspace，其他区域保留未知。[Batch](https://help.aliyun.com/en/model-studio/batch-inference)、[RAG](https://help.aliyun.com/en/model-studio/rag-api-overview) |
| Qwen RAG 服务操作 | 北京 workspace 的 RAG REST 目录分别记录知识库 chunks 的增删改查、data-center 原始文件列表/删除/批量标签更新，以及 storage/QPS 监控。这八项服务级记录只限定 `qwen` 北京 profile，不写入模型能力单元格。[新增 chunk](https://help.aliyun.com/en/model-studio/rag-api-add-chunk)、[列出 chunks](https://help.aliyun.com/en/model-studio/rag-api-list-chunks)、[更新 chunk](https://help.aliyun.com/en/model-studio/rag-api-update-chunk)、[删除 chunks](https://help.aliyun.com/en/model-studio/rag-api-delete-chunk)、[列出原始文件](https://help.aliyun.com/en/model-studio/rag-api-list-file)、[删除原始文件](https://help.aliyun.com/en/model-studio/rag-api-delete-file)、[批量更新文件标签](https://help.aliyun.com/en/model-studio/rag-api-batch-update-tag)、[知识库监控](https://help.aliyun.com/en/model-studio/rag-api-get-index-monitor) |
| Qwen 已发布 Knowledge Search | 北京独立 REST 操作 `POST /api/v1/indices/knowledge/search` 通过已创建并发布的 Knowledge Retrieval 服务检索。调用方传入 `agent_id`，检索策略由已发布服务预先配置。此证据不表示该接口会创建或发布服务，也不同于直接 Knowledge Base 检索。[Knowledge Search API](https://help.aliyun.com/en/model-studio/knowledgesearch) |
| Qwen 已发布 Knowledge Q&A | 北京 workspace 的 `POST /api/v2/apps/knowledge/chat` 接口通过 SSE 流式返回答案。调用要求已创建并发布的 Q&A 服务及其 `agent_id`，每次请求传入完整消息历史，并设置 `stream: true`。接口没有模型选择字段；可选的服务级缓存开关不构成通用模型缓存能力证据。[Knowledge Chat API](https://help.aliyun.com/en/model-studio/knowledgechat) |
| Qwen RAG 数据导入 | 另有类别列出/创建/删除、data connector 创建/查询、从授权 OSS 批量导入文件六项服务级操作，同样只记录北京 `qwen` profile，不加入模型能力单元格。删除类别会使其文件变为未分类；导入 OSS 需要授权的 service-linked role。[列出类别](https://help.aliyun.com/en/model-studio/rag-api-list-category)、[创建类别](https://help.aliyun.com/en/model-studio/rag-api-add-category)、[删除类别](https://help.aliyun.com/en/model-studio/rag-api-delete-category)、[创建 connector](https://help.aliyun.com/en/model-studio/rag-api-add-connector)、[查询 connector](https://help.aliyun.com/en/model-studio/rag-api-get-connector)、[从 OSS 导入](https://help.aliyun.com/en/model-studio/rag-api-oss-import) |
| Qwen 独立实时 ASR | 独立的 Qwen3-ASR-Flash-Realtime WebSocket 通过区域 workspace 主机和 `model` 查询参数接收音频流并返回转写。矩阵只将其限定到北京、新加坡的 Qwen profile，并与文件转写（`service.audio.transcriptions`）、Omni 语音对话（`service.realtime.connect`）和 TTS 分开。稳定模型别名及两个快照仅作为服务模型示例，不写入模型操作单元格；香港和美国 profile 不据此作肯定推断。[连接流程](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-interaction-process)、[客户端事件](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-client-events)、[服务端事件](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-server-events)、[模型](https://help.aliyun.com/en/model-studio/qwen3-asr-flash-realtime)、[区域指南](https://help.aliyun.com/en/model-studio/real-time-speech-recognition-user-guide) |
| Qwen HTTP TTS SSE | Qwen-TTS API 页面同时列出普通响应与 HTTP SSE 输出；SSE 中间块携带 Base64 音频，结束块的 `audio.data` 为空并包含音频 URL。此证据属于 `service.audio.speech`，与 WebSocket 文本合成分开记录。页面的 SSE 请求示例标注新加坡，并注明北京/新加坡 API key 分开。[TTS API](https://help.aliyun.com/en/model-studio/qwen-tts-api) |
| Qwen-TTS-Realtime | 独立文本转语音 WebSocket 使用 `/api-ws/v1/realtime?model=qwen3-tts-flash-realtime`，北京和新加坡示例分别使用各自主机与 API key。矩阵将其记入 `service.audio.speech_streaming_text`，与 `service.realtime.connect` 的语音对话及 ASR 分开；香港和美国 profile 保留未知。`qwen-tts-realtime` 旧系列仅列在中国大陆，`qwen3-tts-flash-realtime` 则列在北京和新加坡。[WebSocket 接入](https://help.aliyun.com/en/model-studio/interactive-process-of-qwen-tts-realtime-synthesis)、[区域模型清单](https://help.aliyun.com/en/model-studio/model-pricing)、[客户端事件](https://help.aliyun.com/en/model-studio/qwen-tts-realtime-client-events)、[服务端事件](https://help.aliyun.com/en/model-studio/qwen-tts-realtime-server-events) |
| Kimi 缓存与工具 | K3 Responses 的缓存是 implicit，TTL 为 5 分钟或 1 小时；显式断点会被拒绝。hosted 工具仅确认 Web Search，其他工具类型被明确排除。[Responses](https://platform.kimi.com/docs/api/responses) |
| Kimi Batch / 退役 | 国际 Batch 明确允许 K2.6、K2.7 Code，明确不允许 K3；旧 K2 和 K2.5 的退役状态单独记录。[Batch](https://platform.kimi.ai/docs/guide/use-batch-api)、[大陆模型目录](https://platform.kimi.com/docs/models)、[国际模型目录](https://platform.kimi.ai/docs/models) |
| MiniMax | Messages 的 `mcp_servers` 被忽略；M2.7 的显式缓存不能从 Anthropic 协议推广到其他模型。普通增量文本 WebSocket TTS 使用 `/ws/v1/t2a_v2`，大陆完整主机来自页面内嵌 AsyncAPI；Bidi TTS 使用独立的 `/ws/v1/t2a_v2_bidi` 路由，具有分块文本、取消、flush、finish 和句子/任务事件。两类 TTS 都与实时语音对话分开记录。[Messages](https://platform.minimax.io/docs/api-reference/text-anthropic-api)、[缓存](https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache)、[语音接口索引](https://platform.minimax.io/docs/llms.txt)、[普通流式 TTS](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket)、[国际 Bidi](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket-bidi)、[大陆 Bidi](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket-bidi) |
| MiniMax 音色服务 | 国际和大陆 API 文档分别明确记录音色复刻（`POST /v1/voice_clone`）、文本描述设计音色（`POST /v1/voice_design`）、查询可用音色（`POST /v1/get_voice`）和删除音色（`POST /v1/delete_voice`）。矩阵只将这些服务记录限定在 MiniMax profile；账户权限仍未验证。[国际复刻](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)、[设计](https://platform.minimax.io/docs/api-reference/voice-design-design)、[查询](https://platform.minimax.io/docs/api-reference/voice-management-get)、[删除](https://platform.minimax.io/docs/api-reference/voice-management-delete)；[大陆复刻](https://platform.minimax.cn/docs/api-reference/voice-cloning-clone)、[设计](https://platform.minimax.cn/docs/api-reference/voice-design-design)、[查询](https://platform.minimax.cn/docs/api-reference/voice-management-get)、[删除](https://platform.minimax.cn/docs/api-reference/voice-management-delete) |
| GLM / Z.AI | JSON Object 不等于原生 JSON Schema。大陆 Batch、异步 Chat、TTS 与国际 ASR 分别取证，Coding Plan 不继承标准 API 服务。[JSON 模式](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output/)、[Batch](https://docs.bigmodel.cn/cn/guide/tools/batch)、[国际 ASR](https://docs.z.ai/api-reference/audio/audio-transcriptions) |
| DeepSeek | JSON Object 和自动缓存有证据；未成功核实的模型/协议/操作保持未知，不因为 OpenAI/Anthropic 兼容就继承其功能。[JSON](https://api-docs.deepseek.com/guides/json_mode/)、[缓存](https://api-docs.deepseek.com/guides/kv_cache/) |

图像模型行在此次七类缺口审计中仅做目录覆盖，生成/编辑单元格保留未知；已有实现及 TOML capability 布尔值不是此次第一方证据。未核实的 Coding Plan 别名、国际服务和 schema 子集也保留未知。

## 维护与验证

先阅读具体厂商、地区、模型和协议的官方契约，再更新单元格、来源日期与 availability。若只有服务级接口证据，只修改服务记录；增加新快照时保留旧来源日期，并确保记录日期不晚于快照。若服务操作只覆盖一组有证据的 profile，使用 `profile_ids` 限定适用范围。目录增加或更名时同步补模型行；依据明确下线公告移除运行目录模型时，将证据移入 `retired_models`。

运行 `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 cargo test --test capability_matrix_china`。测试核对 profile/原始模型/图像模型全覆盖、身份与区域一致、操作集合完整、无重复记录、状态和来源合法，以及 Batch/缓存/实时语音/Coding Plan、Qwen 北京 RAG scope、HTTP TTS SSE、实时 TTS 区域边界、独立实时 ASR scope 和已发布 Knowledge Search/Q&A 前置条件。它不联网、不调用账户，也不会把 mock 或目录覆盖自动标为线上验证。

Qwen Audio Generation 新增独立北京工作空间服务记录，覆盖 `qwen-audio-3.1-tts-next` 的参考音频和签名 URL 输出，不向 Chat 模型外推能力，也不代表账户实测。旧证据保留原核验日期。[官方 API](https://help.aliyun.com/en/model-studio/audio-generation-api)。

LiveTranslate3.5/3.8 新增北京、新加坡工作空间独立服务记录，其会话字段与文本事件分别编码，不与 Omni 对话混用。[官方指南](https://help.aliyun.com/en/model-studio/qwen3-5-livetranslate-flash-realtime)。

当前 GLM Knowledge 文档明确提供 `upload_document/{id}` 文件 multipart 上传，不能与废弃 Agent 上传路由混淆。MiniMax HTTP TTS SSE 按区域单列，并以官方 CLI 解析器补充响应契约证据，不推断未展开的 stream_options 子字段或字幕事件结构。GLM Realtime 官方 SDK 定义了函数输出 call_id，即使旧示例省略该字段也不能据此认定不支持。以上均是文档发现，不是线上验收。
