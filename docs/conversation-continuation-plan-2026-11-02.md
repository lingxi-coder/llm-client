# 多 provider 会话增量续接：文档核查与更新准备

核查日期：2026-11-02（America/Los_Angeles）。代码基线：llm-client `a30a3a9`。
状态：实施准备时的核查快照。后续实现、接口与边界见[会话增量续接与恢复](conversation-continuation.md)；下文的“当前代码”描述准备阶段基线。尚未进行真实账户请求或性能验收。

目标：支持有官方依据的 provider 以“上一轮响应引用 + 新增输入”续接，减少历史上传、服务端解析等开销。宿主仍保存完整、可重放的历史；远端会话不能成为唯一副本。提示缓存、工具容器复用和 SDK 本地 chat 对象不等同于这项能力。

## 1. 官方能力与当前实现

以下结论绑定具体服务端点、协议、区域和账户产品，不由模型名称或兼容格式推导。官方文档确认不等于已通过真实账户验收。

| Provider / 路由 | 官方结论 | 当前代码 | 更新决定 |
| --- | --- | --- | --- |
| OpenAI API Responses | `previous_response_id`；HTTP 存储与 WebSocket 连接内续接需分别处理。[O1] | `ContinuationRef`、Responses 编码及 `ResponsesSession` 增量已存在 | 保留并补查存储、失效恢复、实际启用条件 |
| Anthropic Claude API Messages | Message threads beta；首轮 create、后续 continue。[A1] | 没有类型化 thread、增量状态与重建流程；统一续接未向 Messages 开放 | 优先新增 provider 专属适配 |
| Google Gemini Interactions | `previous_interaction_id`；配置须每轮重传，`store=false` 的结果不能作为后续引用。[G1] | 原生 `InteractionRef` / `previous` 与统一 codec 续接已存在 | 补齐宿主自动增量及恢复策略；不把 generateContent/chat 当作同一服务 |
| xAI Grok Responses HTTP | 支持 `previous_response_id` 与服务端存储。[X1] | `grok-responses` 已声明支持，复用 Responses codec | 纳入统一 session 与恢复验收 |
| xAI Grok Responses WebSocket | 支持连接内增量，包括 `store=false`；断线且无持久状态时需全量恢复。[X2] | 通用 WS 基础存在；预设构造统一设置 `supports_websockets=false` | 核对握手、帧、错误后单独开放路由能力；不能只改续接布尔值 |
| Qwen / 百炼 Responses | `previous_response_id`，响应 ID 有效 7 天；`store` 默认 true；session cache 是独立选项。[Q1][Q2] | `qwen-search{,-intl,-hk,-us}` 是 Responses 路由，但未声明续接支持 | 对已核实区域/端点/模型开放，增加存储校验；不更改 Chat 默认协议 |
| 智谱国内 Responses | 独立 `/api/v1/responses`，默认 `store=false`；显式 `store=true` 后可续接，ID 有效 7 天。[Z1] | 国内 `glm` / `glm-coding` 尚无该 Responses 路由；现有国际 `zai-coding-responses` 关闭续接 | 单独新增国内路由与模型能力适配，不能直接开启国际 Coding 路由 |
| Z.AI 国际 / Coding Plan | 本次查到的国际文档和索引未建立相同的服务端续接合同。[Z2] | `zai-coding-responses` 明确 false | 保持关闭，标记待证实，不宣称厂商全面不支持 |
| DeepSeek Responses | 官方明确无状态，不支持 previous ID/store；不支持的参数可能被静默忽略。[D1] | `deepseek-responses` false | 保持完整历史；加入防止静默丢上下文的回归 |
| MiniMax Responses | 官方明确忽略 `previous_response_id`，响应恒为 null。[M1] | 国内/国际 Responses 均 false | 保持完整历史；缓存能力单独处理 |
| Kimi 平台 Chat / Responses | 多轮指南明确说明无状态与历史回传；本次 Responses 参数页读取失败，不能把 Chat 结论推广至所有 Responses。[K1] | 平台为 Chat；`kimi-code-responses` false | 平台 Chat 保持全量，Responses 续接记待核实 |
| Kimi Code 订阅 | 官方模型配置页不能证明远端 previous ID 续接。[K2] | 专属 Coding 端点且续接 false | 保持关闭，不与平台 API Key、CLI 本地 session 混用 |
| OpenRouter Responses | 官方明确无状态；`store=true` 或非空 previous ID 返回 400。[R1] | 未声明续接能力 | 保持全量；`session_id`/sticky routing 仅用于路由和缓存亲和 |
| GitHub Copilot | SDK 有 session，但该合同不是本项目直连 `api.githubcopilot.com` 的历史省略合同。[C1] | 预设为 Chat | 待直连端点官方证据；不以 SDK 本地 session 推导支持 |

Azure/Foundry、Vertex、Bedrock 等托管服务必须按各自路由另行核对。Anthropic 兼容网关不会自动获得 Claude API 的 Message threads 能力。本轮不新增其他厂商或迁移既有账户产品。

## 2. 代码落点与现有边界

- `src/protocol/continuation.rs`：引用绑定 provider、profile、endpoint fingerprint、account scope、request model、workspace。保留这些隔离约束。
- `src/client/executor.rs:381`：当前统一能力判定只支持 Gemini Interactions，或显式声明 `supports_previous_response_id` 的 Responses。返回引用的逻辑需要补查有效存储策略和完成状态，避免把普通 response ID 当作一定可续接。
- `src/codecs/openai/responses/encode.rs:134`：显式续接编码 previous ID，但仍编码调用方给的全部 messages；HTTP 不会自动替调用方裁剪历史。
- `src/websocket.rs:68`、`src/client/session.rs`：WS 会在完整前缀及非 input 参数匹配时生成增量 wire body；中断后清理状态。它先构造完整 logical request，不能承诺省掉客户端全部历史编码成本。
- `src/presets.rs:334`：构造预设时 WS 支持为 false，库存在实现不代表实际账户默认已启用。
- `src/codecs/anthropic/encode.rs`、`src/protocol/request_controls.rs`：缺少 thread 类型化请求；不能通过泛用 body 注入绕过账户隔离和重试策略。
- `src/codecs/gemini_interactions/encode.rs`、`src/providers/google/interactions/mod.rs`：已有 previous ID、原生存储和引用生命周期能力，应复用。
- `src/wire_options.rs`：已有 previous response ID 保留字段；新增 thread/previous interaction/conversation 等远端状态字段时同步禁止非类型化注入。
- Harness `crates/llm-runtime/src/service.rs:10424` 已在流式调用中复用 ResponsesSession；`client.rs` 依据路由选择 WS。Cargo 当前固定同一完整修订 `a30a3a96907a7e3847af4aa999e7ddf7b4590dcf`；更新应改依赖 pin，不直接修改 `deps/llm-client` 副本。

## 3. 接口与状态设计（待实现）

### 路由能力

新增类型化 continuation capability，分开表达 `unsupported`、`unknown`、Responses 持久续接、Responses 连接内续接、Anthropic thread、Gemini interaction。记录存储要求、传输、作用域和官方来源。不要继续用一个布尔值代表所有行为。

未知/未声明能力默认发送完整历史。显式要求服务端续接而路由不支持时，在发送前返回明确错误；自动模式允许按策略发送全量。兼容端点返回 response ID 不构成支持证据。

### 调用方式

保留现有显式 `ChatRequest.continuation`。新增可选会话 helper，接受宿主提供的完整逻辑历史或可重放快照，管理最后一个完成引用、已接受历史边界、固定参数指纹与连接代次。具体公共类型名在实现时确定。

helper 只处理传输续接，不执行工具、不审批、不持久化凭证。完整历史由宿主持有；helper 可保存受控内存快照或摘要/边界，用于裁剪和恢复。先做正确性，再按测量结果优化避免重复完整编码。

每次准备输出 logical request 与实际 wire request 的摘要，明确 full / delta / recreate。签名、配额与 admission 应针对最终 wire 请求，恢复发送也必须重新经过宿主授权和预算路径。

### 状态迁移

`无引用 → 首轮全量 → 完成响应 → 可续接 → 新增输入`。
历史编辑、摘要压缩、账户/区域/端点/模型切换、协议切换或固定字段变化，按 provider 合同重建。流式取消或不完整响应不能提交为可续接状态；同一 session 的后续轮次串行化，分支使用独立状态。

继续保留所有可重放输出：reasoning、thought signature、provider-native items、tool call/result、phase 及工具收据。不能从 UI 展示文本重建完整历史。远端失效恢复时不重新执行本地工具，只重放已有结果。

### Anthropic 专属规则

按 [A1] 实现 beta header、create 与 continue。continue 省略 system/tools，校验固定字段；只发送新增消息。以 `message_stop` 确认提交引用；支持 pause_turn 的空 messages 续接。

本地完整历史始终保留。thread 404 必须识别 `thread_not_found` 等错误码，不能把所有 404 当作过期。已知可恢复的 thread 失效/竞争错误可在预算内重建一次；重建失败后按明确策略停用 thread 或返回错误。refusal、fallback、thinking block、缓存断点和压缩的处理遵循官方规则，不以通用 Responses 逻辑代替。

线程空闲 5 分钟过期，服务端为 best-effort；本地 TTL 只用于提前选择重建，不证明服务端一定持有状态。thread 与 prompt caching 分开配置。当前官方不支持的 context editing/compaction、HIPAA 等组合需本地拒绝或走全量模式；不静默丢掉用户配置。

### Responses / Gemini 差异

- OpenAI/Qwen：续接时重新发送本轮 instructions；不能依赖上一轮 instructions 继承。
- xAI：分别处理持久引用与当前 socket 内引用；store=false 的连接内引用不能在重连后继续使用。59 分钟连接上限及 previous_response_not_found 纳入恢复。
- 智谱国内：显式存储策略；只为已证明存储的完成响应产生持久引用。新路由还需独立核对工具、模型和推理参数，不能整套套用 OpenAI 能力。
- Gemini：每轮重新发送 tools、system_instruction 和 generation_config；原生 InteractionRef 与统一引用需保持相同账户和生命周期约束。
- Qwen：`x-dashscope-session-cache: enable` 作为独立可选优化；缺失该 header 不代表不支持 previous ID。区域、知识库 workspace 和端点迁移继续保留作用域校验。

### 错误、恢复和可观测性

只有明确未进入推理的可恢复错误才允许自动重建。传输超时、响应途中断开等结果未知场景，不得以“引用失效”为由无条件重复生成或切换 provider。保留现有重试、幂等性和结果未知合同。

记录模式、协议、重建原因、历史边界、新增项数量、logical/wire 字节数与 TTFT。ID、正文和凭证默认不写日志；如需关联使用不可逆摘要。缓存命中与用量只采信上游，不能以请求缩小推断 token 免费或计费降低。

## 4. 分阶段实施与完成条件

1. **公共合同**：能力类型、引用生命周期、自动/显式策略和 provider 专属状态；明确存储关闭、变更和失效行为。先做单轮/流式引用资格回归。
2. **已有路由补齐**：Qwen 已核实区域 Responses 开放续接；OpenAI/xAI/Gemini 接入统一 helper，保留已有 WS 行为。xAI WS 单独验收后开放。
3. **Anthropic**：独立实现线程 create/continue、固定字段指纹、流式提交、重建与不支持组合，不给其他 Messages 网关自动启用。
4. **智谱国内**：新增可选择的 Responses 账户路由与模型适配，沿用原账户产品边界；国际 Z.AI/Coding 待证据后另行开放。
5. **Harness**：在会话生命周期内复用 helper；取消、compact、fork、切模型、切账户均使状态正确失效。发布 llm-client 修订后更新 Cargo pin。
6. **真实验收**：对每个要启用的账户做三轮文本、工具结果、长历史、过期/断线恢复；确认服务端记住首轮随机事实且实际第二轮 body 不含首轮。没有有效凭证的组合保持“未验收”。

必要的 mock/contract 验收包括：

- 首轮 full、后续 delta，logical 上下文与全量基线语义一致；不会重复附加上一轮 assistant 或工具输出。
- account/profile/endpoint/model/workspace 切换拒绝旧引用；HTTP 与 WS 连接内引用不能混用。
- store=false 不产生持久引用；允许连接内续接的 provider 仅在正确连接代次使用。
- Anthropic header、system/tools 省略、固定字段改变、TTL、精确错误分类、重建一次上限、pause_turn、refusal/fallback、缓存断点重放。
- WS 正常结束、取消、断线、重连、连接上限、prewarm 和 HTTP fallback；已有显式分支优先于自动状态。
- DeepSeek/MiniMax/OpenRouter 继续发送完整历史，绝不因响应里出现 id 而裁剪。
- 压缩/编辑历史后重建；reasoning/signature/native items/tool result/phase 无损。
- admission、预算、取消和结果未知不被恢复逻辑绕过。

实施时优先运行 `continuation`、`responses_session`、`gemini_interactions_codec`、`interactions`、`presets` 现有测试，加上 provider 新合同测试；再运行所改协议相关回归。真实性能比较采用相同模型、历史、缓存策略和网络条件，分别报告上传字节、TTFT、端到端耗时与用量，不能把 mock 通过称为加速验收。

## 5. 官方来源与证据限制

- [O1] [OpenAI conversation state](https://developers.openai.com/api/docs/guides/conversation-state)。本会话前一轮已读取官方正文。
- [A1] [Anthropic Message threads](https://platform.claude.com/docs/en/build-with-claude/message-threads)。本会话前一轮已读取官方正文。
- [G1] [Gemini Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)。本轮通过官方文档的 Context7 索引读取存储与续接条款；直接网页读取超时/重定向失败，实际发布前复核当前 schema。
- [X1] [xAI Responses 与 Chat 比较](https://docs.x.ai/developers/model-capabilities/text/comparison)。
- [X2] [xAI WebSocket mode](https://docs.x.ai/developers/advanced-api-usage/websocket-mode)。
- [Q1] [百炼 Responses 参数](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-responses)。
- [Q2] [百炼 Responses 指南](https://www.alibabacloud.com/help/en/model-studio/compatibility-with-openai-responses-api)。
- [Z1] [智谱国内 Response API 兼容](https://docs.bigmodel.cn/cn/guide/develop/responses/introduction)。注意这是 `bigmodel.cn` 官方文档，与第三方 `bigmodel.io` 无关。
- [Z2] [Z.AI Chat Completion](https://docs.z.ai/api-reference/llm/chat-completion)、[国际文档索引](https://docs.z.ai/llms.txt)。本次未找到对应续接合同，非否定性证明。
- [D1] [DeepSeek Responses 支持矩阵](https://api-docs.deepseek.com/guides/responses_api/)。
- [M1] [MiniMax Create Response](https://platform.minimax.io/docs/api-reference/responses-create)。
- [K1] [Kimi 多轮对话](https://platform.kimi.com/docs/guide/engage-in-multi-turn-conversations-using-kimi-api)。[Responses 参数页](https://platform.kimi.com/docs/api/responses) 本次多次读取失败，该协议保留待核实。
- [K2] [Kimi Code 模型配置](https://www.kimi.com/code/docs/en/kimi-code/models)。
- [R1] [OpenRouter Responses overview](https://openrouter.ai/docs/api_reference/responses/overview)。API 生成参考页虽列出 previous ID 字段，以明确说明无状态的行为指南为准。
- [C1] [GitHub Copilot SDK quickstart](https://docs.github.com/en/copilot/get-started/sdk-quickstart)。SDK session 仅作为单独产品合同，不用于证明本项目直连 API 的续接支持。

准备阶段仅修改本文档。后续实施记录见功能文档与对应测试；本核查快照不作为实现验收证明。
