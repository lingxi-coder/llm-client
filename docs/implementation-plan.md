# 全能力规划实施状态

这是已批准规划的持续实施清单。第二十六轮已完成当前已确认功能清单的实现与本地统一验收；这不代表所有提供方、账户和区域均已在线验证。不保留旧 API、旧配置或自动迁移分支。客户端负责协议、传输、资源生命周期；工具执行、权限、历史、凭证刷新、压缩与 RAG 编排属于宿主。

当前状态以文末第二十六轮记录为准；前面各轮“待测试”“下一轮”等表述保留为历史过程，不作为当前待办。未确认的官方契约、真实账户验收和宿主职责分别列出，不折算成编码任务数量。

## 已落地的部分

- [x] 0.2.0 公共 Chat 类型与服务入口；移除旧顶层 Chat 入口和类型名，更新调用方、示例及双语文档。
- [x] 配置版本 v3，拒绝旧文件且不改写；独立 Embedding 服务的继承/替换/禁用持久化。
- [x] 结构化输出四种基础协议的编码、原生字段冲突检查、JSON/schema/反序列化结果校验、保留失败响应。
- [x] 结构化流式最终结果便捷接口；终止后校验，失败保留事件和已重建的响应。
- [x] 提示缓存请求级策略、Messages 自动缓存和工具/system/消息断点、TTL 和无效组合校验。
- [x] 文本 Embeddings 的 OpenAI-compatible / Gemini / Qwen 编解码、独立 endpoint/auth、deadline、结果完整性校验。
- [x] OpenRouter Embeddings 使用独立适配器，明确映射检索 query/document 的 `input_type`、维度和浮点向量；其他 task 发送前拒绝。
- [x] OpenRouter Embedding 专属模型目录：独立 endpoint/auth、offset/limit 分页、原生元数据保留及异常分页拒绝。
- [x] Responses 托管工具及未知原生输出项的非流式/流式保留与去重、等待审批状态。
- [x] Responses 带 provider/profile/端点/模型/账户/工作空间绑定的续传引用；流式终止后可提取，作用域不符在发送前拒绝，续传请求不故障转移或自动重提。
- [x] 普通函数 `tools` 与提供方执行的 `hosted_tools` 分开；原 Web Search 与 Qwen File Search 共用托管工具列表，重复声明在副作用前拒绝。
- [x] OpenAI Responses 的 Code Interpreter 自动容器请求与 1g/4g/16g/64g 档位，原生调用项和执行输出保留。
- [x] OpenAI Vector Stores 独立检索路由、作用域绑定的 store/file/batch 引用、文件属性写入/更新与筛选、自定义分块、批量索引/取消/状态及文件分页、第一页语义搜索、排名和查询重写、资源删除；提交结果未知时显式返回，不自动重提。
- [x] OpenAI Batch 基础服务：purpose=batch 文件上传、JSONL/custom_id 编码及 writer、大输入 multipart 流式上传、独立路由、作用域绑定的提交/查询/列举/取消、结果/错误文件引用；64 MiB 内整文件读取和大结果文件逐行流式读取；未知提交结果不重提。
- [x] xAI Deferred Chat：独立提交/结果路由，作用域绑定句柄、`202` 归还句柄、`200` 单次消费，原生结果保留；未知提交/读取结果不自动重试。
- [x] OpenAI Responses Background：独立配置路由、`background=true` 提交、作用域绑定句柄、状态查询与取消；终态和原生错误保留，未知提交/取消结果不自动重试。
- [x] OpenAI Responses Background 流式恢复：`background=true,stream=true` 原生 SSE 事件、递增 `sequence_number` 游标、断线后 `starting_after` 查询；未获任务 ID 前中断显式标记结果未知。
- [x] OpenAI Audio 文件转写/翻译：独立服务路由、25 MB 内定长 multipart 流式上传、Whisper 时间戳与 diarized JSON 说话人结果、模型/输出组合发送前校验。
- [x] OpenAI 文件转写 SSE：一次性流式上传、原生增量/说话人/完成事件、缺失终态与中断错误，不自动重提。
- [x] OpenAI TTS 基础二进制流：独立 speech 路由、GPT 与旧 TTS 模型/音色限制、六种输出格式、PCM 采样元数据、分块读取和中断字节数。
- [x] Gemini Interactions 基础服务：独立路由、模型/Deep Research agent 文本创建、账户绑定的 `previous_interaction_id` 与 GET、原生步骤及状态保留；`stream=true` 原生 SSE 步骤事件。
- [x] Gemini Interactions 断线流式恢复、取消与删除；恢复引用绑定账户和端点，沿用 provider 的事件游标。
- [x] Anthropic Messages 托管工具搜索的原生工具定义、延迟加载和回放，以及 Anthropic Batch 的独立提交/查询/列举/取消与逐项结果流。
- [x] Gemini Embedding 2 多模态输入、Gemini File Search 的 store/导入操作生命周期；GLM 知识库创建、URL 文档导入与检索。
- [x] MiniMax ASR、OpenRouter 音频转写/合成、Qwen 文件异步转写；后者具备显式区域/工作空间绑定和单次任务查询。
- [x] Qwen、Kimi、OpenRouter 的原生 Batch 切片，以及 GLM 异步 Chat 任务；各自保留独立 wire 和任务作用域。
- [x] xAI Collections 的管理 key/API key 分离、文件与搜索操作；Realtime 的可注入传输、OpenAI 基础事件编解码、有界队列和帧限制。
- [x] Qwen 北京工作空间原生知识库检索及文档/索引任务管理；新加坡 REST 路由文档未确认，因此发送前拒绝。
- [x] xAI 原生 REST STT/TTS：multipart 转写、原始二进制合成流，以及带时间戳的 JSON 音频结果；显式账户/路由和逐次凭证。
- [x] xAI 原生 Batch 的先创建后添加请求、任务状态、请求元数据、分页结果和取消；Qwen 非实时 TTS 返回短时效音频 URL，下载由调用方决定。
- [x] GLM-ASR 的自托管 SGLang Chat Completions 路径，显式路由和账户作用域；不等同于智谱托管云端 ASR API。
- [x] MiniMax 原生同步及异步长文本 TTS：`.io`/`.cn` 显式账户路由、同步 Hex 音频或短时效 URL、异步提交与查询；独立 WebSocket TTS 与双向 TTS 见下方完成项。
- [x] Qwen `qwen3.7-text-rerank` 原生重排，采用文档明确列出的北京工作空间路由并保留原始文档索引与分数。
- [x] Gemini Developer API inline 与文件式 Batch：`generateContent` 批次创建/查询/列举/取消、独立文件上传与结果流；不将其误接到 Interactions API。
- [x] OpenRouter 原生文本 Rerank，保留原始候选索引、相关度分数、provider 与 usage；与 Embeddings 和知识库资源服务分开。
- [x] GLM 大陆地区原生 Batch 的输入上传、任务提交/查询/取消及结果流；国际地区无已确认路由时发送前拒绝。
- [x] OpenAI Batch 的调用方持久文件引用清单与提交前元数据/过期校验；保留文件生命周期所有权在调用方，不在后台任务期间自动清理。
- [x] OpenAI 显式容器创建/挂载已有 Files API 文件时的逐文件预检与到期余量校验；请求只发送一次。
- [x] OpenAI Background 完成/不完整事件中的完整 Response 重建；失败事件保留原生游标和错误，不把未确认的 REST 取消事件当成终态。
- [x] OpenRouter Chat 音频输入与原生流式音频 delta 保留；输入格式、角色、模型 modality 和输出流式前置校验。
- [x] GLM 云端 ASR（大陆及国际）与大陆 TTS 独立服务；国际 TTS 无确认路由时前置拒绝。
- [x] Gemini Interactions 单次及 SSE 流式语音合成：3.8 Flash/Flash-Lite TTS、独立账户路由、格式校验、原生事件/音频增量、作用域绑定游标与中断状态。
- [x] OpenRouter 网关响应缓存的请求级策略与明确 HIT/MISS 响应头投影；流式及非流式均不从零用量猜测命中。
- [x] xAI Realtime 工具结果批量提交与显式继续；支持并行 call_id 结果先提交、由宿主决定继续时机。
- [x] GLM 大陆 Realtime 独立 WebSocket 协议：会话确认、音频与文本输入、音频/转写/用量事件；国际端点未确认时不猜测。
- [x] xAI Responses 原生 Remote MCP：独立配置、逐次 Secret、原生工具调用保留；拒绝 xAI 未支持的 OpenAI 审批/connector 参数。
- [x] MiniMax 国际/大陆 WebSocket 流式 TTS：持续文本输入、独立音频事件和中断状态；两地区完整主机已依据官方内嵌 AsyncAPI 确认。
- [x] Qwen Chat 显式消息缓存断点：五分钟/最多四处，按模型、区域与部署 scope 校验，缓存写入用量在同步/流式均计入。
- [x] Qwen Responses 托管 Code Interpreter：仅已确认模型/路由，保留原生调用状态和 `x_tools.code_interpreter.count`；不推断工具费用。
- [x] Qwen Omni Realtime：北京与新加坡已确认 WebSocket 路由、音频先行的 JPEG 帧输入、音视频/转写事件；其他协议明确拒绝图像帧。
- [x] OpenAI Speech 可使用已获批自定义音色 ID：`voice` 对象原生编码、仅 GPT TTS 模型、无副作用 ID 预检。
- [x] OpenAI 自定义音色资源：同意短语目录、同意录音 CRUD 与音色创建；同意引用绑定 profile、端点和调用方账户范围，上传中断返回未知结果。
- [x] OpenRouter 官方 Responses 与 Anthropic Messages 路由的网关响应缓存：精确 endpoint 预检、请求头与 HIT/MISS 投影；Messages codec 不再重复拼接 `/v1`。
- [x] Qwen Responses Web Extractor：与 Web Search 配套的工具声明、原生调用项和独立调用次数；目前限定已确认的模型。
- [x] xAI Collections 文档重新索引：作用域绑定引用、Management API key 与原生 PATCH；任务后续状态由调用方查询。
- [x] Claude Messages 严格工具与 JSON 输出的请求级 schema 编译上限预检：20 个严格工具、24 个可选参数、16 个 union 参数；不删改 schema。
- [x] Claude 原生 JSON 输出与 citations、assistant prefill 的组合预检，以及严格工具/输出 schema 的递归引用、allOf 引用和 format 限制；共享引用遍历去重。
- [x] OpenAI Speech SSE：类型化音频增量、完成事件与原始未知事件；完成用量校验、有界分帧、断流错误及终态后停止读取。
- [x] Anthropic 与 Gemini Batch 删除；Qwen、Kimi、GLM Batch 分页列举，保留各自分页结构、作用域与异常响应处理。
- [x] 普通文件的一次性流式上传：各现有 multipart adapter 与 Gemini resumable 路径、声明长度校验、不自动轮询、结构化未知上传结果；MiniMax 异步 TTS 输入上传前要求账户作用域。
- [x] OpenAI 自定义音色作用域引用：创建结果与显式本地导入返回绑定 provider/profile/Speech endpoint/account 的引用；持久化保留范围，线上请求只发送音色 ID。
- [x] Gemini 异步 Embedding Batch：独立操作引用、当前 embedContentConfig、内联及 typed JSONL 输入、逐项响应/错误、文件结果增量解码，以及查询/列举/取消/删除；按官方输入顺序校验已知 key、维度和结果数。

- [x] xAI 双向流式 TTS、多轮合成/取消/替换表更新，以及 Custom Voices 创建、分页列举、详情、更新、删除和参考音频读取；REST TTS 语速、延迟优化、规范化与替换表。

- [x] MiniMax 音色复刻、设计、类别列举和删除；大陆/国际文件上传作用域与音色引用安全接入 HTTP、异步及国际/大陆 WebSocket TTS。
- [x] MiniMax 国际/大陆独立双向 TTS：服务端分句、flush/cancel/finish、软错误、显式 Ping、取消与提前断流后的终态处理。
- [x] Qwen HTTP TTS SSE 与北京/新加坡独立实时 TTS、独立实时 ASR；保留各自事件、音频缓冲及完成/关闭契约，不从 Omni 推断能力。
- [x] Qwen 北京知识库分块、分类、connector、OSS 导入和监控操作，以及已发布 Knowledge Search/Chat 服务；Chat SSE 保留调用方历史、原生阶段和托管工具结果。
- [x] Anthropic 第一方 Code Execution 的类型化声明、作用域容器/文件续用及同步/流式顶层容器 metadata。
- [x] 文件已知到期与处理状态从上传/查询引用保留到模型输入，并在派发前预检；Gemini `PROCESSING`/`ACTIVE`/`FAILED` 与 Qwen `uploaded`/`processing`/`processed`/`error` 按各自协议处理，未知状态不跨 provider 推断。

- [x] DeepSeek 第一方 Chat JSON 模式的提示词预检，以及 Qwen/GLM 官方 Embedding 模型的维度与批量限制；不扩展到未知模型或普通兼容网关。
- [x] OpenAI Responses 原生 Tool Search、延迟函数/MCP 加载、客户端执行交接与原生回放，以及请求级提示缓存 key、模式、断点和独立的 retention 配置。
- [x] Anthropic Programmatic Tool Calling 的声明、组合预检、调用者信息保留及待处理调用续接；客户端不执行宿主工具。
- [x] CI 增加全特性检查，覆盖可选 WebSocket 后端；保留默认依赖隔离与独立 tokenizer 编译。
- [x] xAI Collections 配置更新：作用域引用、Management API PUT、字段预检及响应 ID 核对；单次派发且不自动重试。
- [x] Anthropic Code Execution 容器 Skills 声明：内置及已上传自定义引用、版本、20 项上限与工作空间作用域；独立 Skills 资源管理见第十五轮。
- [x] 共享 Realtime 初始化全部帧在连接前生成及校验，非法配置不建立远端连接或部分发送。
- [x] xAI Files 50 MB 上传预检、Collections 独立 100 MB 流式上传与批量文档 metadata 查询，以及上传结果到普通文件服务的安全引用转换。
- [x] Anthropic Skills 独立资源及版本 CRUD、流式上传、显式内容 ZIP 下载；工作空间引用接续到 Messages，保留额外来源类型与未知执行结果。

以上为阶段内的可用切片，不代表 P0–P7 任何整阶段已全部完成。具体接口和限制见 [服务说明](services.md)。

## 原始范围的当前覆盖与剩余边界（第二十五轮核对）

1. **P0 能力矩阵与协议：**三份逐操作证据矩阵覆盖当前全部 29 个内置 profile、559 个模型行，并将 10 个已下线模型单列历史证据：[OpenAI/Gemini](capability-matrix-openai-gemini.md)、[中国提供方](capability-matrix-china.md)、[其他提供方](capability-matrix-west.md)。这只是文档证据，不代表模型在每个账户/区域可用或真实调用通过；大量能力单元格仍是 Unknown；Unknown 是证据状态，不逐格等同编码任务。已确认的有限实施清单与本轮验收见文末。Gemini Interactions 创建/查询/续接、原生流式步骤、断线恢复、取消、删除及多模态/工具请求编码已落地，Gemini 3 与 Remote MCP 的官方不支持组合已在发送前拒绝。初始来源索引见 `data/capability-audit.json`。
2. **P1 公共架构：**所有服务配置独立化；已允许显式禁用 Chat 且不声明 Chat 模型的独立服务 profile；公共服务配置与原生服务显式 scope 均独立于 Chat，见下方逐域审计；普通文件及现有 multipart ASR 入口已有流式路径，Base64 JSON 保留协议所需缓冲；继续逐域核验资源引用绑定 provider/profile/服务/endpoint/account/model。独立可选 Rustls WebSocket transport 已落地，仍需其他 provider 与真实端到端验收。
3. **P2 输出与缓存：**补全各 provider/model 的 schema 子集与组合限制。Qwen 第一方 Chat 的 JSON 提示预检与已验证模型的可选属性规则已接入；Qwen Chat 的显式消息缓存断点与写入用量已落地，OpenAI Responses 已补齐 key、模式、30 分钟最小 TTL、内容断点及独立的最大 retention 参数；其他独立协议的缓存仍待核实；OpenRouter Chat 已增加按模型限制的 Anthropic、Alibaba/Qwen、OpenAI GPT-5.6+ 显式 prompt 缓存映射，其他网关模型仍待核实。Gemini Developer API 已有独立的显式缓存资源 CRUD、账户/端点作用域和逐次凭证；其他远端缓存提供方仍待确认与实现。OpenRouter 官方 Chat、Responses、Messages 与 Embeddings 已有独立的网关响应缓存请求策略及明确响应头的命中状态投影；其他远端缓存服务仍待核实。
4. **P3 Agent 工具：**Anthropic 工具搜索/defer_loading、第一方 Code Execution 与作用域容器/文件续用及 Programmatic Tool Calling，OpenAI Responses 原生 Tool Search、OpenAI Responses 的 Remote MCP 配置/逐次凭证/审批请求与回应、xAI Responses 的独立 Remote MCP 配置/逐次凭证、Qwen Responses Code Interpreter、Web Extractor 及 `web_search` 的 `open_page` 原生结果已落地；Qwen 官方未确认独立 `web_fetch` 工具类型。Anthropic 第一方 Web Fetch 的四个版本、URL 来源策略、缓存/strict 组合校验及原生结果/用量已接入；Anthropic Messages MCP 和中途 System/消息级 effort/clear_at 也已落地，第一方及 Vertex Browser/Computer 稳定客户端工具集与命名空间往返已接入；Vertex Tool Search、System/消息级控制与引用型工具增删已接入。Foundry 部署身份、Tool Search、分托管 Web Fetch 和 MCP 2025 beta 本轮已接入。内联 strict/PTC 与 typed Code Execution/Web Fetch 放置已补齐；Foundry Anthropic 托管的执行/容器、Files 和 custom Skills 已接入。完整顶层 deferred catalog 与中途引用增添的组合已覆盖；仅消息内按值定义是否被 Tool Search 索引仍缺明确契约，不推测支持。xAI 不支持的 OpenAI MCP 审批与 connector 参数发送前拒绝。OpenRouter Responses/Messages 的托管正则工具搜索、Shell、延迟函数加载和有作用域的容器引用已接入；其他提供方托管工具、更多 MCP 组合及所有协议完整原生事件回放仍待完成。OpenAI 显式容器及文件创建、列举、详情、下载、删除、已有文件挂载与到期预检已落地；来源文件到期后容器副本的保留语义未获官方确认，不作保证。还需继续审计其他有状态工具与异步提交的自动重试/切换语义。
5. **P4 知识库：**OpenAI 检索索引文件列表已支持状态过滤、创建时间排序与前后游标；搜索响应虽有 `next_page`，当前官方 OpenAPI 与 SDK 请求均无 cursor/next_page；保留原生响应，不虚构续页请求，此项不列作已确认编码缺口。真实账户验收尚未执行。Gemini File Search 的当前官方 store/document/import/upload 生命周期已实现，新增删除账户作用域与 `force` 语义回归；GLM 知识库已有 CRUD、文档列举和 URL 导入；按当前官方契约新增 `upload_document/{id}` multipart 文件上传、逐文件部分结果与未知提交状态，未采用已废弃 Agent 上传接口。Qwen 北京工作空间已增加文档/索引任务、分块增删改查、数据中心文件列举/删除/标签更新、分类管理、connector 创建/查询、授权 OSS 导入、知识库监控及原生检索/重排，其他区域未核实的 REST 路由仍拒绝；xAI Collections 已增加文档索引重建、检索模式与列表过滤/排序，OpenRouter Rerank 也有独立切片。第二十五轮对照现有 Gemini File Search、xAI Collections 和 Qwen DashScope RAG 生命周期，未发现这些已审计接口的已确认缺失操作；Alibaba 另一个签名 OpenAPI 管理接口不混入 DashScope bearer 路由。Embedding 多模态的 Gemini Embedding 2 已落地，OpenRouter、Gemini 与 OpenAI 独立模型目录已接入，OpenAI Embeddings 的按模型参数和输入数量限制已补齐；未公开的参数范围保持提供方负责，不从默认值推断上限。
6. **P5 Batch / Background：**OpenAI Batch 已有调用方持久文件引用清单、提交前文件元数据与过期校验；真实账户验收及跨作业结束后的文件清理策略仍归调用方。Anthropic、Gemini inline 与文件式、Qwen、Kimi、OpenRouter、xAI、GLM 大陆地区的独立 Batch 切片与 GLM 异步 Chat 已落地，已审计的现有 Batch 服务公开生命周期均已有对应入口；新服务或新操作必须先确认独立契约后计入实施清单。Qwen Batch 错误文件、OpenRouter Batch 日期/端点专属模态、xAI 文件式 Batch 已补齐。OpenAI Background 已对完整终态事件重建中立响应并支持显式删除及不确定删除结果，仍需真实账户验收；xAI Deferred 需真实账户验收及宿主跨进程单次消费协调。未知提交结果禁止自动重提。
7. **P6 Audio：**其他提供方已公开的 ASR/翻译/TTS；更多 Chat 音频模型/格式、采样率与真实账户验收。OpenAI TTS SSE 已实现；文本对齐属于等待官方公布事件/时间戳契约的项目，不能据猜测增加字段。OpenAI 基础转写/翻译、转写 SSE、已知说话人参考音频、二进制 TTS、自定义音色 ID 合成及同意录音/音色创建生命周期，以及 Gemini Interactions 单次/流式 TTS、MiniMax ASR/同步/异步长文本/国际与大陆普通及双向 WebSocket TTS/音色复刻与设计管理、OpenRouter 独立 Audio 与 Chat 音频、Qwen 异步 ASR/HTTP TTS 与 SSE 音频输出/独立实时 TTS 与 ASR、xAI REST 文件/URL STT、TTS 与音色列表、独立实时 STT WebSocket、流式 TTS WebSocket、自定义音色完整生命周期、GLM 自托管 ASR 和 GLM 云端 ASR/大陆 TTS 的分项切片已落地。Qwen 独立 Audio Generation、Audio 3.x 文件 ASR、MiniMax ASR SSE/HTTP TTS SSE 与字幕参数、OpenRouter TTS 参考音频已补齐。MiniMax 大陆 WSS 主机已从官方内嵌 AsyncAPI 确认并接入；GLM 国际 TTS 官方路由仍未确认，因此发送前拒绝。
8. **P7 Realtime：**可注入传输、OpenAI 基础事件编解码、有界队列和帧限制、可选的内置 Rustls WebSocket 后端，以及 Gemini Live、xAI Voice、GLM 大陆、Qwen Omni 北京/新加坡的会话设置/输入输出驱动已落地。xAI、OpenAI 与 GLM 工具结果可批量提交，并由宿主显式继续；OpenAI Realtime 图像/音色、独立 GPT-Live 主 WebSocket 与 Qwen 3.5/3.8 LiveTranslate 已补齐。端到端真实服务端恢复及账户/区域验收仍待完成。不含设备层或 WebRTC。
9. **全量交付：**补齐跨 provider fixtures 与无副作用 preflight 测试、配置事务、限流/超时/取消/错误体、未知用量/价格；fmt/test/clippy/rustdoc/downstream/package；所有真实账户验收需独立记录，不把 mock 成功标成线上验证。

## 验收要求

### 2026-09-26 逐操作收尾审计

能力矩阵中的 `supported` 是厂商文档结论，必须另行对照公共服务方法才能判定实现完成。本轮从已确认的服务操作反查代码，继续追踪以下具体缺口；它们没有替代上面的 P0–P7 原始范围：

- 已实现 Anthropic Batch 删除：包括账户作用域预检、确认对象校验及删除失败语义。
- 已实现 Gemini Batch 删除：删除操作记录不等于取消任务，使用独立 DELETE 接口。
- 已实现 GLM、Qwen、Kimi Batch 分页列举：各服务分别遵守厂商自己的分页与返回结构，拒绝重复记录与无法前进的分页。
- 已实现 OpenAI Speech SSE 音频增量和完成事件；文本对齐与未文档化错误字段等待官方公开契约，未知事件原样保留。
- 已实现 Claude 严格工具和 JSON 输出的递归引用、allOf 引用、format 限制，以及 citations、assistant prefill 组合预检；其他 provider/model 的 schema 子集审计仍保留原始范围。
- 已实现 Gemini 异步 Embedding Batch：按官方 [Batch REST 参考](https://ai.google.dev/api/batch-api) 的 `models.asyncBatchEmbedContent` 独立编码输入与结果；持久化引用区分操作类型，文件结果按 [Embedding 输出契约](https://ai.google.dev/api/embeddings#EmbedContentBatchOutput) 的输入顺序校验，关联 key 冲突不会被静默接受。
- 已实现 P1 文件流式上传：新增 `UploadFileStream` 与 `FileService::upload_stream`，覆盖现有文件上传 adapter，并与 `Bytes` 便捷入口共用 multipart 字段编码；大文件可选择新流式入口避免整文件再次缓冲。
- 已统一 MiniMax 异步 TTS 输入作用域：`FilePurpose::AsyncTtsInput` 的缓冲及流式上传均在上传副作用前检查非空账户作用域。
- 已实现自定义音色引用：`CustomVoiceRef` 绑定 provider/profile/Speech endpoint/account；创建及显式本地导入后才能交给 `SpeechVoice::Custom`，发送前校验范围。导入不查询或证明服务端授权。
- 已复核 Gemini 同步 Embedding 配置：编码已使用当前官方 [Embedding REST 参考](https://ai.google.dev/api/embeddings) 的 embedContentConfig，并为已确认模型关闭自动截断；异步 Batch 应沿用当前契约，不增加旧版顶层参数回退。
- 已实现 Gemini Batch 更新：按当前 [Batch REST 参考](https://ai.google.dev/api/batch-api) 提供 `batches.updateGenerateContentBatch`，编码直接 GenerateContentBatch 资源、类型化更新掩码和字符串形式的 int64 priority；校验最终请求大小、操作类型与返回引用。定向测试 29 项通过，当前轮集成验证另行记录。
- P0 新增能力证据已回填：Gemini 矩阵分别列出生成批次与 Embedding 批次提交，依据模型官方文档确认 embedding-001 与 embedding-2 的支持；其他模型保留 Unknown。补充批次结果服务操作，所有真实账户验证状态仍为 `not_run`。当前轮矩阵定向测试 6 项通过；真实账户状态未变。
- P1 独立配置审计：公共服务使用 profile 的 `ServiceSetting`；Gemini Batch/Context Cache、xAI Collections、Qwen Knowledge 等原生服务使用显式的独立 endpoint/region/workspace scope，已与 Chat 路由分开。Containers 使用独立的官方固定根地址和账户 scope。缺少统一 profile 开关本身不构成独立路由缺口；为所有低层服务增设相同字段属于额外 facade 设计，不作为本轮必做任务。作用域、逐次凭证及具体服务的校验仍按各自契约验收。
- 已实现 Gemini GenerateContent 托管工具：第一方 Developer API 的明确模型列表支持 Code Execution、URL Context 和 Maps，包含 Maps 经纬度/文本输入输出约束及 Gemini 3 已确认的函数组合；原生代码、内联图片、服务端 toolCall/toolResponse 保留回放，候选级归因 metadata 独立保留。未知模型/网关/Vertex 组合发送前拒绝，托管执行不会自动重试或切换；9 项定向测试通过。多个新内建工具之间未确认的组合仍拒绝，不从单项支持推定可组合。
- 已补齐 Anthropic 原生内容保留：完整未知内容块及带引用的非流式文本保留为 `ProviderContent`，原生服务端工具参数在块结束后组装；未知流事件/delta 通过独立 `ProviderEvent` 保留，不冒充可回放内容或宿主工具。无法解释的 delta、无效或非对象参数不会产生伪完整回放块。流式文本及引用由 codec 在块结束时自动重建，覆盖中途出现引用、非 Web 引用和初始文本；未知或畸形增量保留原始事件并抑制完整回放。未闭合内容块和 8 MiB 聚合缓冲上限均有校验。这不是所有工具参数的类型化适配。原生内容测试 13 项通过，相关已有协议测试保持通过。
- 已补齐 Qwen 数据中心文件：显式申请上传租约、一次性定长 OSS PUT、addFile 注册和单次文件详情查询；租约及文件引用绑定工作空间/账户/区域，签名上传头保持原样且不发送工作空间凭证。已知会话临时文件不能通过类型化辅助入口导入长期知识库。文件流程 9 项及原知识库 12 项定向测试通过；未进行真实账户调用。
- 已补齐 OpenRouter multipart STT 流式上传：沿用现有 `AudioInput`，按已声明大小增量发送，避免完整音频缓冲及 multipart 二次复制；输入长度不符或源流报错时不发送结束边界，派发后的源错误报告未知结果。Base64 JSON 路径仍有界缓冲。原有与新增测试共 11 项通过，独立审查未发现剩余缺陷。

后续验收必须检查实际入口、请求体、响应解码、终态/中断行为及双语示例；真实账户调用仍单独标记，不从本地测试推定。

每个声明完成的能力必须具备可调用 API、正确编码和解析、生命周期及失败语义、测试和双语使用说明。未实现、文档未确认和未做实测分别记录。已有图片生成/编辑/任务服务、文件、计价和 tokenizer 保留回归覆盖。所有当前提供方均在范围内，不能用通用透传或仅返回 HTTP 200 宣称已适配。

## 当前切片验证记录

2026-09-24，本地验证通过：

- `cargo fmt --all -- --check` 与 `git diff --check`。
- `CARGO_INCREMENTAL=0 cargo test --locked --offline --no-fail-fast`：621 项通过。
- `CARGO_INCREMENTAL=0 cargo test --locked --offline --all-features --no-fail-fast`：625 项通过。
- 下游独立 crate 的文档/示例测试：60 项通过，锁文件已更新。
- `cargo clippy --locked --offline --all-targets --all-features -- -D warnings`。
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --offline --all-features --no-deps`。
- `cargo package --locked --offline --allow-dirty`：打包并验证成功，238 个文件、压缩后 9.5 MiB。

测试使用 mock 与本地 HTTP server；因沙箱禁止回环监听，全量测试在获准的本地监听环境运行。编译期间遇到磁盘不足，清理了本项目可重建的 `target/debug/incremental` 后重新完成验证。没有调用付费模型，没有执行发布或 Git 提交。

续传切片的最终验证：默认特性测试 624 项、全功能测试 628 项、下游文档测试 60 项通过；严格 Clippy、Rustdoc、格式、diff 检查及打包验证通过。没有进行真实账户续传验收。

HostedTool 切片的最终验证：默认特性测试 626 项、全功能测试 630 项、下游文档测试 60 项通过；严格 Clippy、Rustdoc、格式、diff 检查及打包验证通过。新增测试覆盖重复工具声明的无副作用拒绝，以及两种托管工具共存时单项替换和旧请求字段拒绝。没有进行真实账户验收。

OpenAI Code Interpreter 自动容器切片的最终验证：默认特性测试 628 项、全功能测试 632 项、下游文档测试 60 项通过；严格 Clippy、Rustdoc、diff 检查及打包验证通过，最终执行了格式修正。新增测试覆盖档位、执行输出请求、原生调用项、未声明适配器的拒绝。没有进行真实账户验收。

OpenAI 托管检索切片的最终验证：默认特性测试 634 项、全功能测试 638 项、下游文档测试 62 项通过；严格 Clippy、Rustdoc、格式、diff 检查及打包验证通过。新增测试覆盖 store/file 生命周期、索引未就绪与就绪状态、搜索来源、账户/文件作用域、配置继承、提交结果未知和只读错误分类。未运行真实账户验收。

OpenAI 检索高级搜索切片的最终验证：默认特性测试 637 项、全功能测试 641 项、下游文档测试 64 项通过；严格 Clippy、Rustdoc、格式、diff 检查及打包验证通过。新增测试覆盖文件属性写入、复合筛选、排名阈值和查询重写的请求编码，以及无副作用参数拒绝。官方搜索请求未文档化后续页游标参数，因此后续页仍未实现；未运行真实账户验收。

OpenAI 检索批量索引与结构化流式终值切片的最终验证：默认特性测试 645 项、全功能测试 649 项、下游文档测试 66 项通过；严格 Clippy、Rustdoc、格式、diff 检查及打包验证通过。新增测试覆盖批次创建/查询/文件分页/取消、上传文件作用域与分块校验、提交结果未知、索引文件列表，以及分片 JSON 的最终校验和失败数据保留。没有进行真实账户验收。

OpenAI Batch 生命周期切片的最终验证：默认特性测试 652 项、全功能测试 656 项、下游文档测试 68 项通过；严格 Clippy、Rustdoc、格式、diff 检查及打包验证通过。新增测试覆盖 JSONL 的 custom_id/模型/流式约束、batch purpose 上传、提交/查询/列举/取消、结果和错误行解析、路由/账户作用域与未知提交结果。结果文件目前有 64 MiB 内存上限，输入 multipart 仍在内存中构造；尚未运行真实账户验收。

Batch 大结果流式读取切片：`stream_result()` 逐行解析、支持超过 64 MiB 的文件，对单行设 16 MiB 上限，并限制最多 50,000 个唯一 `custom_id`。增加拆块、重复 ID、跨账户拒绝和大文件测试；旧 `read_result()` 保持 64 MiB 整文件上限。真实账户验收仍未完成。

OpenRouter Embeddings 切片：内置路由改用独立 `OpenRouter` 适配器，检索 query/document 对应 `search_query` / `search_document`，拒绝未实现的 task。另接入官方 Embedding 模型目录的独立路由、分页和原生元数据。覆盖请求编码、响应索引/用量、目录分页及内置 preset；多模态输入、其他提供方模型目录和真实账户验收仍未完成。

以上切片合并后的最终验证：默认特性测试 686 项、全功能测试 690 项、下游文档测试 71 项通过；严格 Clippy、Rustdoc、格式、diff 检查及离线打包验证通过（269 个文件，压缩后约 9.5 MiB）。测试仅使用 mock 和本地 HTTP server；尚无真实提供方账户验收。验证时同一工作区的快照/缓存重构由另一会话并行完成，最终数值反映当时的合并工作树。

Batch 大输入流式上传切片：新增 `write_jsonl()` 的先校验后写入、`HttpStreamRequest` 与无缓冲回退的 `Transport::send_stream()`、内置 reqwest 定长请求流、`FileService::upload_batch_stream()` 的准确字节数/账户作用域/总超时检查。默认特性测试 692 项、全功能测试 695 项、下游文档测试 71 项通过；严格 Clippy、Rustdoc、格式、diff 与离线打包验证通过（270 个文件，压缩后约 9.6 MiB）。测试覆盖 65 MiB 分块上传、长度不足/超出、鉴权、短超时及本地 HTTP 实际请求；未调用真实提供方账户。Batch 附件完整任务存续期、其他提供方 Batch/Background 与真实账户验收仍待完成。

xAI Deferred Chat 切片：`grok` Chat profile 增加独立 Deferred 提交/结果路由；`ChatRequest` 走所选 Chat codec 后发送 `deferred=true`。句柄绑定 provider、profile、双端点、账户与模型；`fetch_once()` 在 `202` 时归还句柄，在 `200` 时返回解析及原生结果而不归还句柄。未知提交、已接受但缺 ID、读取中断和已消耗但解析失败分别处理且不自动重试；目录模型变更不妨碍读取已排队结果。配置 v3 继承/禁用、作用域及状态测试通过。最终默认特性测试 700 项、全功能测试 704 项、下游文档测试 73 项通过；严格 Clippy、Rustdoc、格式、diff 与离线打包验证通过（274 个文件，压缩后约 9.6 MiB）。未调用真实 xAI 账户；跨进程单次消费协调仍归宿主或后续持久化设计。

OpenAI Responses Background 切片：内置 `openai` profile 增加独立后台路由；使用所选 Responses codec 编码并加入 `background=true`，提供一次提交、查询与取消。句柄绑定 provider、profile、端点、账户及模型；查询/取消和续传输入在发送前校验。`queued/in_progress` 与四种终态分别处理，完成结果提供解码响应和同账户续传引用，原生响应及失败错误保留；未知提交与取消结果不自动重试。配置 v3 继承/禁用、状态/作用域/错误契约测试通过。最终默认特性测试 707 项、全功能测试 711 项、下游文档测试 75 项通过；严格 Clippy、Rustdoc、格式、diff 与离线打包验证通过（278 个文件，压缩后约 9.6 MiB）。没有调用真实 OpenAI 账户；后台流式恢复和事件序号游标仍待实现。

上述验收针对 Background 切片完成时的共享工作树。之后并行编辑曾使附件链和 Gemini codec 的全目标编译短暂失败；以下新一轮合并工作树验收已经重新通过。

OpenAI Background 原生流式恢复切片：新增 `submit_stream()` 与 `resume_stream()`，首个含 response ID 的事件建立任务引用，之后每个完整 SSE 事件返回可序列化的账户/端点绑定游标；恢复请求使用 `stream=true&starting_after=<sequence_number>`。序号必须递增，断线和非终态 EOF 归还最近游标；错误 HTTP 状态不会伪装为 SSE。默认特性测试 745 项、全功能测试 754 项、下游文档测试 77 项通过；严格 Clippy、Rustdoc、格式、diff 与离线打包验证通过（288 个文件，压缩后约 9.6 MiB）。覆盖事件分块、断线恢复、账户作用域、序号倒退、缺失任务 ID 和 HTTP 错误；没有调用真实 OpenAI 账户。流式接口保留原生事件，不从中途游标重建先前内容的 provider-neutral 响应。

OpenAI Audio 文件转写/翻译切片：新增独立 `client.audio()` 服务和 v3 路由继承/禁用，定长 multipart 流式上传 1–25,000,000 字节文件；支持已核实的转写模型与 Whisper 英文翻译，并在发送前校验输出格式、时间戳、diarize 选项和文件信息。JSON 结果保留原生响应，类型化返回单一语言/检测语言数组、时长、词/段时间戳及说话人。默认特性测试 755 项、全功能测试 763 项、下游文档测试 79 项通过；严格 Clippy、Rustdoc、格式、diff 与离线打包验证通过（301 个文件，压缩后约 9.6 MiB）。测试覆盖请求流准确长度、无缓冲回退、模型限制、错误体与配置持久化；没有调用真实 OpenAI 账户。文件转写 SSE、已知说话人参考、TTS、Chat 音频及其他提供方仍待实现。

OpenAI TTS 二进制流切片：`client.audio().synthesize()` 使用独立 speech 路由和 JSON 请求，支持四个官方 TTS 模型、模型适用的内置音色、六种输出格式、可选指令及 0.25–4.0 语速。`SpeechStream` 不缓冲整段输出，PCM 标明 24 kHz、单声道、16 位，断流错误报告已交付字节数，成功但无音频字节时显式报错；副作用未知时不自动重提。默认特性测试 763 项、全功能测试 772 项、下游文档测试 81 项通过；严格 Clippy、Rustdoc、格式、diff 与离线打包验证通过（300 个文件，压缩后约 9.6 MiB）。测试覆盖请求编码、旧模型限制、错误体和二进制分块；未调用真实账户。TTS 的 SSE 文本对齐、自定义音色、其他提供方和 Chat/Realtime 音频仍待完成。

OpenAI 文件转写 SSE 切片：`client.audio().transcribe_stream()` 使用独立转写路由的一次性定长 multipart 上传并发送 `stream=true`，逐个归还原生 delta、segment 和 done 事件；仅 done 是成功终态，缺失终态或无效事件明确报错。Whisper 与非 JSON 输出发送前拒绝；HTTP 错误保留状态、请求 ID 和原生错误体，不自动重提或尝试恢复。默认特性测试 767 项、全功能测试 776 项、下游文档测试 83 项通过；严格 Clippy、Rustdoc、格式、diff 与离线打包验证通过（301 个文件，压缩后约 9.6 MiB）。测试覆盖原生事件、说话人字段、缺失终态、无效 SSE、Whisper 限制和 HTTP 错误；未调用真实账户。

Gemini Interactions 基础切片：内置 `gemini` profile 增加独立 `/v1beta/interactions` 路由，`client.interactions()` 支持文本模型和 Deep Research agent 创建、`previous_interaction_id` 续接、账户/路由绑定引用的 GET 查询，以及 `stream=true` 原生 SSE 生命周期与步骤事件。缺失 `[DONE]`、无效事件和 interaction ID 变化显式报错；保留 `event_id` 和原生 JSON，但不猜测断线恢复参数。默认特性测试 774 项、全功能测试 783 项、下游文档测试 85 项通过；严格 Clippy、Rustdoc、格式、diff 和离线打包验证通过（306 个文件，压缩后约 9.6 MiB）。测试覆盖配置继承/禁用、模型和 agent 请求、跨账户拒绝、HTTP 错误、SSE 终止与中断；未调用真实 Google 账户。断线恢复、取消/删除、多模态/工具和全 provider 能力矩阵仍待完成。

2026-09-25 并行 provider 切片合并验收：默认特性与全特性 `cargo test --locked --offline --quiet --no-fail-fast` 均通过；全目标/全特性严格 Clippy、`RUSTDOCFLAGS='-D warnings' cargo doc --locked --offline --all-features --no-deps`、下游文档测试（115 项通过、22 项标记忽略）、`git diff --check` 和 `cargo package --locked --offline --allow-dirty` 通过。打包包含 402 个文件，压缩后 9.8 MiB。测试使用 mock 和本地回环服务器，没有调用真实 provider 或付费推理。中央入口、本轮新模块及非冲突文件已按仓库的 Rust 2021 edition 定向运行 rustfmt；完整 `cargo fmt --all -- --check` 仍对 6 个跨会话共用的 codec/协议文件报告格式差异，本会话保留这些文件的当前编辑状态，以免干扰另一会话的快照/缓存重构。磁盘空间紧张时仅清理了本项目 `target/debug/examples` 可重建产物，未删除源码、持久数据或发布产物。

2026-09-25 本轮合并工作树验收：最终默认特性测试 1,096 项通过、全特性测试 1,114 项通过，均 0 失败，各有 6 项示意文档代码忽略；下游独立 crate 138 项通过、48 项示意代码忽略。全目标/全特性严格 Clippy、`RUSTDOCFLAGS='-D warnings' cargo doc --locked --offline --all-features --no-deps`、`cargo fmt --all -- --check`、`git diff --check` 与离线 `cargo package --locked --offline --allow-dirty` 均通过。打包包含 482 个文件，压缩后约 10 MiB。需要本地回环监听的现有 HTTP 测试在允许监听的环境运行；未调用真实 provider 或付费推理。三份证据矩阵与当前 559 个模型目录行完全对齐，10 个官方已下线模型单列历史记录；各服务和模型仍需逐账户/区域验收，大量模型操作保持 Unknown。磁盘紧张时仅清理本项目可重建的旧 `target/debug/deps` 对象文件和测试执行文件，未删除源码、持久数据或发布产物。

2026-09-25 最新合并工作树验收：Qwen Omni Realtime、Qwen Responses Code Interpreter、OpenRouter Chat 按模型提示缓存，以及 OpenRouter Embeddings 网关响应缓存均有可调用接口、原生编解码/响应投影、无副作用预检、回归测试与双语文档。默认特性测试 1,117 项、全特性测试 1,135 项通过，均 0 失败、6 项示意文档代码忽略；下游独立 crate 139 项通过、52 项示意代码忽略。全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式检查、`git diff --check` 及离线打包验证通过；包内 495 个文件，压缩后 10.3 MiB。测试使用 mock 与本地回环服务；未调用真实 provider 或付费推理。P0–P7 的未完成项仍以上述清单为准，尤其逐账户/区域真实验收与大量 Unknown 能力单元格，不能由本地测试替代。

2026-09-26 合并工作树验收：本轮新增 OpenRouter 官方 Responses/Messages 缓存路由、Qwen Web Extractor、xAI Collections 文档重建索引、OpenAI Speech 自定义音色 ID 合成及同意录音/音色创建资源生命周期。默认特性测试 1,132 项、全特性测试 1,151 项通过，均 0 失败、6 项示意文档代码忽略；随后修复两处严格 Clippy 报告的局部问题，受影响音色测试再次通过。下游独立 crate 141 项通过、54 项示意代码忽略。全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式及 diff 检查、离线打包验证通过；包内 503 个文件，压缩后 10.3 MiB。测试只使用 mock 和本地回环服务，未调用真实 provider 或付费推理。OpenAI 音色权限与项目归属仍需真实账户验收；P0–P7 的其他未完成项继续以上述清单为准。

2026-09-26 后续合并工作树验收：Claude 原生 Messages 的严格工具/JSON 输出共用 schema 上限发送前校验已加入；OpenRouter 等只复用 Messages wire 的网关不继承该限制。Gemini File Search 的官方 store/document 生命周期核对已覆盖现有实现，新增删除账户作用域与 `force` 行为回归。GLM 个人知识库现行官方文档未提供本地文件上传契约，旧版智能体上传接口未接入；OpenAI Speech SSE 已公布成功事件，但流内错误和文字对齐字段仍无完整官方契约。默认特性测试 1,135 项、全特性测试 1,153 项通过，均 0 失败、6 项示意文档代码忽略；下游独立 crate 141 项通过、56 项示意代码忽略。严格 Clippy、警告视为错误的 Rustdoc、格式/diff 检查和离线打包验证通过；包内 504 个文件，压缩后 10.3 MiB。没有真实 provider 调用或付费推理。

## 2026-09-26 第二轮收尾验收

本轮完成 Claude 原生 JSON/严格工具的组合与 schema 预检、OpenAI Speech SSE、Anthropic/Gemini Batch 删除及 GLM/Qwen/Kimi Batch 分页列举。编码由 GPT-6 Luna（max）并行完成，集成审阅修正了共享引用重复遍历、分页一致性、GLM 查询参数 URL 构建及失败语义。没有修改快照/缓存重构。

- 默认特性全量回归：1159 项通过、0 失败、6 项忽略。
- 全功能全量回归：1177 项通过、0 失败、6 项忽略。
- 全量回归后按 Clippy 建议移除 GLM/Qwen 无效克隆及 Speech 测试多余宏，三个受影响目标已再次通过；严格 Clippy 全目标/全功能通过。
- 严格 Rustdoc 通过；独立下游双语文档示例 143 项通过、56 项忽略。
- `cargo fmt --all -- --check`、`git diff --check` 通过。
- `cargo package --locked --offline --allow-dirty` 打包与包内编译通过：504 个文件，17.4 MiB，压缩后 10.3 MiB。

测试包含 mock 与本地回环服务；没有真实账户调用、付费调用、发布或 Git 提交。原始 P0–P7 目标尚未全部完成；下一轮从上方逐操作审计中的文件流、MiniMax 输入作用域、自定义音色引用及 Gemini 异步 Embedding Batch 继续，并保留未文档化接口和真实账户验收的独立状态。

## 2026-09-26 第三轮收尾验收

本轮由 GPT-6 Luna（max）并行完成普通文件流式上传、MiniMax 异步 TTS 输入账户预检、OpenAI 自定义音色作用域引用及 Gemini 异步 Embedding Batch。Gemini 增加内联和 typed JSONL 输入、独立操作引用、已知模型参数限制、完整查询/列举/取消/删除及有界逐行结果解码。结果关联按官方输入顺序检查 key、维度与结果数；冲突 key、超长行和不完整结果会报错。两个独立审阅发现的关联问题已修复并复核。

- 默认特性全量回归：1188 项通过、0 失败、6 项忽略。
- 全功能全量回归：1206 项通过、0 失败、6 项忽略。
- 全量回归后按 Clippy 建议对私有文件输入枚举装箱，公共 API 不变；Gemini Batch 的 24 项测试已再次通过。
- 严格 Clippy 全目标/全功能及严格 Rustdoc 通过。
- 独立下游双语文档示例：151 项通过、56 项忽略。
- `cargo fmt --all -- --check` 和 `git diff --check` 通过。
- `cargo package --locked --offline --allow-dirty` 打包及包内编译通过：506 个文件，17.6 MiB，压缩后 10.3 MiB。

本轮使用 mock 与本地回环测试，没有真实账户调用、发布或 Git 提交。原始 P0–P7 仍未全部完成；下一轮继续核实 Gemini Batch 更新操作、补齐新增操作/模型的证据矩阵，以及尚未完成的其他提供方契约与真实账户验收。

## 2026-09-26 第四轮收尾验收

本轮由 GPT-6 Luna（max）并行完成 Gemini Batch 更新与 Embedding Batch 证据回填、Qwen 数据中心文件上传/注册、Gemini GenerateContent 托管工具、Anthropic 原生内容与流式引用重建，以及 OpenRouter multipart STT 流式上传。Gemini 原生服务端工具与内联图片结果保留回放，Maps/URL 归因 metadata 单独保留，托管执行禁止自动重试或切换。Anthropic 已知引用由 codec 组装为完整原生文本块，未知事件保留为观察数据，不交给宿主执行。

- 最终默认特性全量回归：1231 项通过、0 失败、6 项忽略。
- 全功能全量回归最初为 1248 项通过、1 项失败、6 项忽略；唯一失败为 Bedrock 测试帧缺少 `content_block_stop`。补齐测试帧后，相关 `families` 全功能目标 36 项全部复跑通过；没有为迁就测试放宽解码器完整性要求。
- Gemini 托管工具 9 项、Gemini Batch 29 项、Qwen 文件/知识库 21 项、Anthropic 原生内容 13 项、OpenRouter 音频 11 项定向测试通过；其余原有协议测试保持通过。
- 严格全目标/全功能 Clippy、严格 Rustdoc、格式及 diff 检查通过。
- 独立下游双语文档示例：155 项通过、56 项忽略。
- 离线打包及包内编译通过：513 个文件，17.8 MiB，压缩后 10.4 MiB。

测试使用 mock 与本地回环服务，没有真实账户或付费模型调用，没有发布或 Git 提交。快照/缓存重构未修改。独立配置审计确认：已提供显式 endpoint/region/workspace scope 的原生服务不需要仅为形式统一重复增加 profile 字段。

下一项已确认的开发缺口是 Gemini Embedding 独立模型目录：当前 `EmbeddingService::list_models` 与 `models_endpoint` 仅支持 OpenRouter。需依据 Gemini 自身分页与 `supportedGenerationMethods` 契约实现，不将目录发现写入 Chat 模型目录；其他提供方 schema/工具组合仍按明确官方证据逐项审计。原始 P0–P7 的未知契约与真实账户验收继续单列，不能据本轮本地验收宣布全阶段完成。

## 2026-09-26 第五轮定向验收

已接入 Gemini Embedding 独立模型目录：类型化 list/get、原生分页 token 与原查询范围绑定、按 `embedContent` 明确方法筛选、精确资源 ID 与 baseModelId 分开保留。空筛选页仍保留 continuation，不自动翻页，不推断未提供的模型参数。Google Models API 的两个服务操作已加入证据矩阵，OpenAI/Gemini 矩阵现有 93 个来源、34 项服务操作；模型支持单元格没有因目录接入而被提升。

Qwen 第一方 Chat 的 JSON Object 提示要求在附件解析、认证与 HTTP 之前检查，支持紧邻中文文本的大小写混合 JSON。当前明确验证的 `qwen3.8-flash` / `qwen3.8-max` 严格输出允许可选属性和原始 `additionalProperties`，不沿用 OpenAI 的两项专有约束；其他现有 schema 子集检查仍保留。这不改变函数工具、Qwen Responses 或通用网关的规则，也不把客户端验证范围描述为厂商全部支持范围。

- 定向测试 47 项全部通过：证据矩阵 7、Embedding 10、Gemini 目录 10、结构化输出 20。
- 格式与 diff 检查通过。新增双语示例已纳入现有 downstream 文档入口；完整集成检查将在下一轮源代码冻结后统一运行。
- 没有真实账户调用、发布或 Git 提交。

从三份矩阵反查公共 API，又确认 xAI 实时 STT、xAI TTS 音色列表、OpenRouter Responses/Messages server tool search 与 shell 尚无对应入口。下一轮按各自官方契约实现；这三项有明确证据，区别于未公开契约、真实账户验收和宿主策略。

### 服务级矩阵入口复核

静态复核统计了三份矩阵的 `services` 数组：共 160 条 `supported` 记录、95 个不同 operation ID，其中中国提供方 90 条 / 25 个操作，OpenAI/Gemini 34 条 / 34 个操作，其他提供方 36 条 / 36 个操作。审计时发现缺少入口的是 xAI STT WebSocket、TTS voices，以及 OpenRouter Messages/Responses 各自的 shell/tool_search，共 6 个 operation ID；voices 已有定向测试 4 项通过，原 xAI 音频 6 项也通过。

除这些正在实现的操作，其他服务级操作均找到公开调用入口。另有已实现操作的输入模式缺口：xAI REST STT 官方支持 `file` 或服务端下载的 `url`，现有入口仅接受文件；本轮继续补 URL 模式。Qwen 同步 ASR 可通过公开 Chat API 的 `ContentBlock::Audio` 与显式 `extra.body.asr_options` 调用，不因缺少专用门面重复建设服务。

此复核不等同于完成所有模型级能力组合、真实账户验收或矩阵以外的新操作。xAI TTS WebSocket、自定义音色生命周期等尚不在本次矩阵服务集合中；需独立契约与证据条目，不能混入“已验收”结论。原 P0–P7 广义范围继续保留。

## 2026-09-26 第六轮收尾实现

本轮已接入上述 6 个服务操作及 xAI REST STT URL 输入：

- xAI 音色列表：一次 GET，类型化 ID/名称/可选语言及完整原生 metadata，保持来源作用域；4 项定向测试通过。
- xAI URL 转写：独立 URL 类型和 `transcribe_url`，将 URL 原样放入 multipart，下载由 xAI 服务端执行；普通 HTTP transport 即可使用。复用转写参数、响应解析与未知派发结果规则；URL Debug 隐藏地址，错误仅清理明确的逐字回显，不声称识别供应商任意改写。4 项定向测试通过，原文件上传/合成 6 项测试通过。
- xAI STT WebSocket：固定官方地址、原生 query 和二进制音频、等待 ready、显式 finalize/audio.done、有界队列及帧、连接超时；每通道唯一且有效的 final 都到达且 audio.done 实际发出后，才把真正 EOF 视为正常完成。畸形事件保留原文，传输失败不伪装成功，不自动重连或重放。12 项定向测试通过。
- OpenRouter Tool Search / Shell：Responses 与 Messages 原生工具声明、延迟加载、Shell 参数和有作用域的容器引用；直接编码器与高层客户端都拒绝不支持的协议，账户不匹配在附件解析/认证/HTTP 之前失败。服务端调用保留为原生回放数据，不交给宿主执行，自动重试/切换关闭。Messages 非流式容器 envelope 原样保留；流式容器信息通过 ProviderEvent 保留。10 项定向测试通过，结构化输出 20 项及协议族 36 项回归通过。

OpenRouter `allowed_tools` 的完整跨协议 wire schema 尚未获得公开依据，因此工具搜索与延迟加载组合目前要求 Auto；`openrouter:bash` 的客户端/服务器执行语义不同，不混入 shell。本轮只完成已确认的具体操作，真实账户与原始广义能力矩阵中的 Unknown 继续单列。本轮统一验证已完成，结果如下。


### 第五、六轮最终验证

- 默认特性全量回归：1274 项通过、0 失败、6 项忽略。此后补齐了 OpenRouter 容器作用域尾斜杠规范化及持久化回归，并在最终全功能门禁中重新覆盖。
- 最终全功能全量回归：1293 项通过、0 失败、6 项忽略，包含最新 OpenRouter 10 项、xAI STT 12 项及 Gemini 目录 10 项。
- 严格 Clippy 全目标/全功能、严格 Rustdoc、格式与 diff 检查通过。
- 独立下游双语文档示例：163 项通过、0 失败、56 项忽略；新增示例均作为可编译样例检查，未发起真实调用。
- 离线打包及包内编译通过：527 个文件，18.0 MiB，压缩后 10.4 MiB。

本轮已关闭本次服务矩阵审计明确找到的 6 个操作入口缺口及 xAI URL 输入模式缺口。没有真实账户调用、发布或 Git 提交，也没有重做快照/缓存架构。原 P0–P7 不标记全完成：模型级 Unknown、未公开契约、真实账户验收，以及尚未纳入当前服务矩阵的音频扩展仍需分别推进。

## 2026-09-26 第七轮：xAI 音频扩展

按原 P6 继续覆盖已公开但尚未进入服务矩阵的操作，本轮已实现以下切片：

- 独立 TTS WebSocket：固定 `/v1/tts` WSS 路由、文本增量、逐轮音频结果、取消、多轮复用和发音替换表更新；与 STT 和语音对话的完成条件分开。
- Custom Voices：创建、分页列举、详情、元数据更新、删除及显式流式读取参考音频，共六项官方操作；资源引用绑定配置/端点/账户，创建及更新不自动重试。
- 现有 REST TTS 参数：语速、延迟优化、文本规范化、发音替换表。替换表的基本限制供 REST/WSS 复用；不模拟服务端的跨片段匹配和扩展后长度计算。
- 补齐七项新服务操作的第一方证据；模型支持单元格不随服务入口扩展而提升，账户/地域及真实调用状态保持独立。

依据 [xAI TTS](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech) 和 [Custom Voices](https://docs.x.ai/developers/model-capabilities/audio/custom-voices)。Custom Voices API 的 Enterprise/地域限制由提供方账户权限判定；没有公开的 consent 字段或额外下载路由时不自行创造。西方证据矩阵现有 45 项服务操作、50 个来源和 89 个操作定义；模型行与单元格未提升。

流式 TTS 的完成与取消事件按命令实际发送门控，取消确认会清理旧轮次计数；释放最后一个 control 会关闭连接。自定义音色的写入回执不合法时保留请求 ID/原始响应，下载错误或 EOF 会立即释放底层响应流，包装对象仍存活也不会继续占用读取资源。

### 第七轮验证

- 定向测试：REST TTS 8、URL 转写 4、自定义音色 11、流式 TTS 8、西方矩阵 7，共 38 项通过。
- 全功能回归 1314 项通过、0 失败、6 项忽略；随后增加的下载流即时释放修复和回归又通过全功能定向测试 11 项。
- 下游双语文档示例 167 项通过、0 失败、56 项忽略。
- 最终默认特性回归 1297 项通过、0 失败、6 项忽略，包含下载流即时释放回归。
- 全目标/全功能严格 Clippy、严格 Rustdoc、格式与 diff 检查通过。
- 离线打包及包内编译通过：535 个文件，18.2 MiB，压缩后 10.4 MiB。最终 Clippy 清理后再次运行音频定向测试，自定义音色 11 项和流式 TTS 8 项全部通过。

本轮没有真实账户调用、发布或 Git 提交。未知模型组合、未公开契约与账户/地域验收仍按原计划单列。OpenAI TTS SSE 文本对齐经再次核对没有公开事件/时间戳契约，明确列为等待厂商契约，不伪造可执行实现项。


## 2026-09-26 第八轮：MiniMax 音色生命周期

从已公开但未有公共入口的音频操作继续推进，本轮已实现：

- 独立音色服务：复刻、文本设计音色、按类别列举、删除。大陆和国际分别使用已公开的 API 路由与账户作用域。
- 复用 `FileService` 的 VoiceClone / PromptAudio 文件引用，补齐 `api.minimax.cn` 官方文件路由识别；不重复建设上传服务。
- 给现有 HTTP TTS、异步 TTS 和国际 WebSocket TTS 增加显式音色引用消费入口，在请求前验证 provider/profile/account/region/endpoint；内置音色的原始 ID 入口继续可用。
- 请求与结果保留提供方原生字段；复刻/设计/删除结果未知时不自动重提，试听结果原样返回，不自动下载或激活音色。
- 增加四项服务操作的大陆/国际独立证据、契约测试和双语示例。

依据 MiniMax 官方 [Voice Clone](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)、[Voice Design](https://platform.minimax.io/docs/api-reference/voice-design-design)、[Get Voice](https://platform.minimax.io/docs/api-reference/voice-management-get)、[Delete Voice](https://platform.minimax.io/docs/api-reference/voice-management-delete) 及其大陆文档。大陆 Voice Design 的 `aigc_watermark` 独立建模，国际显式设置会在发送前拒绝；普通文本保留换行。复刻成功回执不要求不存在于官方契约的 `voice_id`，使用请求 ID 建立引用。异步 TTS 操作 URL 清理尾斜杠，原有文件/任务作用域指纹保持原值。

### 第八轮验证

- 定向测试 63 项通过：音色生命周期 12、HTTP TTS 8、异步 TTS 13、WSS TTS 9、中国文件 5、流式上传 10、中国矩阵 6。
- 最终全功能回归 1333 项通过、默认特性回归 1315 项通过，均 0 失败、6 项忽略。
- 严格全目标/全功能 Clippy、严格 Rustdoc、格式与 diff 检查通过。
- 下游双语示例 171 项通过、0 失败、56 项忽略。
- 离线打包与包内编译通过：539 个文件，18.3 MiB，压缩后 10.5 MiB。
- 中国矩阵新增八条地区服务记录，共 578 条服务记录；21 个 profile、66 个模型和 1,254 个模型单元格保持不变。
- 没有真实账户调用、发布或 Git 提交。


### 第八轮记录的下一项缺口：MiniMax 双向 TTS（已在第九轮实现）

官方索引将普通 WebSocket TTS 与 `speech-t2a-websocket-bidi` 分列。已读取的[大陆 Bidi 文档](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket-bidi)明确使用独立 `/ws/v1/t2a_v2_bidi` 路径，由服务端累积文本并分句；当前普通 `minimax_streaming_tts` 的逐段完成计数不能代替它。尚需独立的 cancel/flush、sentence_start/sentence_end/task_canceled/task_flushed、会话 ID 和软错误语义。原生未知事件虽已保留，不能据此声称支持 Bidi 生命周期。

国际[文档索引](https://platform.minimax.io/docs/llms.txt)列出同名接口，但本轮尚未直接取得国际 Bidi 完整端点契约。下一轮先核实完整 WSS 主机与地区，再实现；不会仅从普通 TTS 或 HTTP 主机推断新路由。


## 2026-09-26 第九轮：MiniMax 双向 TTS 与完整 WSS 契约

本轮直接读取官方 HTML 内嵌的 `apiReferenceData.channel`，补齐纯文本文档提取器未显示的 `servers` 信息：国际主机为 `api.minimax.io`，大陆为 `api.minimax.cn`，协议均为 `wss`。普通路径 `/ws/v1/t2a_v2` 与双向路径 `/ws/v1/t2a_v2_bidi` 分开。第八轮结束时的“完整端点未确认”状态因此已解决。

已实现并通过本地全量验收：

- 独立双向 TTS 服务，服务端分句、可选 session_id、flush/cancel/finish 各自的确认和结束语义；2204/2205 保持会话并交由宿主决定是否重发。
- 两地区普通 WSS 完整请求参数与 8 款模型，依据当前内嵌 schema；双向复用同契约参数类型，不复用普通协议的逐段完成计数。
- 普通大陆 WSS 路由和带作用域的音色引用消费，国际/大陆引用发送前隔离。
- 显式 WebSocket Ping 能力，由调用方调度；不自动重连或生成保活 JSON。非正常 Close 状态码作为传输错误返回，不丢弃为普通 EOF。
- 两条 Bidi 服务证据与普通大陆路由证据更新；模型单元格不因此提升，真实账户状态仍未验证。

权威来源为对应官方页面内嵌 AsyncAPI：[国际 Bidi](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket-bidi)、[大陆 Bidi](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket-bidi)、[国际普通 WSS](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket)、[大陆普通 WSS](https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket)。

本轮生命周期回归另外覆盖：发送 future 被取消后会话永久关闭并唤醒等待读取；提前 EOF 不恢复为可写状态；终止 `task_failed` 主动关闭连接；发送返回前到达的控制确认不会提前开放输入。原始音频帧无 `event` 字段时仍按官方示例解码，显式未知事件继续原样保留。

GLM 国际 TTS 于本轮再次核实：[Z.AI 官方索引](https://docs.z.ai/llms.txt)与[OpenAPI](https://docs.z.ai/openapi.json)仍只公开音频转写路由，未公开 `audio/speech`。该项继续等待官方契约，不从大陆主机推断国际路径。

### 第九轮验证

- 全特性：1,359 项通过，0 失败，6 项示意文档代码忽略。
- 默认特性：1,338 项通过，0 失败，6 项示意文档代码忽略。
- 独立下游 crate：173 项文档测试及 1 项单元测试通过，56 项示意代码忽略。
- 全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式及 diff 检查通过。
- 离线打包及包内编译验证通过：543 个文件，18.5 MiB，压缩后 10.5 MiB。
- 中国提供方矩阵现有 580 条服务记录；21 个 profile、66 个模型和 1,254 个模型能力单元格保持原范围。

验收日志位于 `/private/tmp/llm-client-{allfeatures,default,rustdoc,downstream,package}-20260926i.log`；最终严格 Clippy 日志为 `/private/tmp/llm-client-clippy-final2-20260926i.log`。所有回归使用 mock 或本地回环服务，没有真实账户调用、付费调用、发布或 Git 提交。P0–P7 的未知模型组合、未公开契约、其余逐操作审计和真实账户验收仍未全部完成；本轮未修改快照/缓存重构。

## 2026-09-26 第十轮：Qwen 知识库资源与 HTTP TTS SSE

依据当前[官方 RAG API 目录](https://help.aliyun.com/en/model-studio/rag-api-overview)，现有知识库/文档/导入任务与上传流程之外还有已公开操作。本轮补齐分块创建、分页列举、更新、批量删除；数据中心文件按游标列举、删除、批量标签更新；知识库监控读取。地域沿用官方北京工作空间路由，资源绑定账户、profile、workspace 和所属知识库/文档。写操作不自动重提，业务失败不能被 HTTP 200 掩盖。

并行审计另外确认两项后续实现缺口：

- 第一方 Anthropic Code Execution 的类型化工具声明和容器续用；当前原生内容回放不足以构成完整支持，同步与流式顶层容器 metadata 还需补齐。依据[Code Execution](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)与[Messages reference](https://platform.claude.com/docs/en/api/typescript/messages)。
- Qwen 独立 TTS Realtime 文本缓冲/音频输出协议；现有非实时 HTTP TTS 与 Omni 会话不能替代独立 TTS 协议。北京/新加坡独立路由及事件见[连接指南](https://help.aliyun.com/en/model-studio/interactive-process-of-qwen-tts-realtime-synthesis)与[客户端事件](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-client-events)。

本轮本地全量验收已完成；上述后续项目未标记完成。

Qwen 音频审计还确认 HTTP TTS SSE 与独立 ASR Realtime 缺口。本轮并行补 HTTP TTS SSE；独立 ASR Realtime 按[连接指南](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-interaction-process)和[客户端事件](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-client-events)留待后续实现，不能用 Omni 转写事件替代。

同一 Qwen RAG 官方目录另列分类列举/创建/删除、OSS 导入和 connector 创建/查询；当前仍需逐页核实并实现，不因本轮补齐 8 个操作而宣称 Qwen RAG 全覆盖。托管 Knowledge Chat/Search 与 Agent 管理应另行确认协议服务边界，不能引入客户端内的宿主 RAG 编排。

本轮实现已接入：Qwen 知识库分块 4 个操作、数据中心文件 3 个操作、监控读取，以及完整文本请求的 HTTP TTS SSE。针对 Update/Delete/Monitoring 官方成功样例省略 `success` 的情况，增加限定操作的响应策略；只有 `code=Success` 且 `status_code=200` 才接受该样例，其他操作保持原契约。TTS 完成必须有合法 stop 块及干净 EOF；late error、提前断流、HTTP 408 和原生错误分别保留正确的失败/未知状态。

### 第十轮验证

- 全特性：1,399 项通过，0 失败，6 项示意文档代码忽略。
- 默认特性：1,378 项通过，0 失败，6 项示意文档代码忽略。
- 独立下游 crate：181 项文档测试及 1 项单元测试通过，56 项示意代码忽略。
- 全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式及 diff 检查通过。
- 离线打包及包内编译验证通过：553 个文件，18.6 MiB，压缩后 10.5 MiB。
- 中国提供方矩阵：588 条服务记录，62 个引用来源；模型能力单元格未因新增服务而提升。

验收日志位于 `/private/tmp/llm-client-{allfeatures,default,rustdoc,downstream,package}-20260926j.log`；最终严格 Clippy 日志为 `/private/tmp/llm-client-clippy-final2-20260926j.log`。本轮新增 40 项回归测试，仅使用 mock 和本地回环服务；没有真实账户调用、付费调用、发布或 Git 提交，也没有修改快照/缓存重构。后续继续上方已确认的 Qwen 独立实时语音、Anthropic Code Execution 和其他知识库操作，原始 P0–P7 尚未全部完成。

## 2026-09-26 第十一轮：Qwen 实时 TTS、知识库导入资源与 Anthropic Code Execution

正在并行实现：

- Qwen 独立实时 TTS：北京/新加坡官方 WSS、模型与音色类别设置、append/commit/clear/finish、音频及 response/session 事件。区别于 Omni 和 HTTP TTS SSE；缓冲清除不冒充取消在途合成，终态与异常断流分别处理。
- Qwen 数据中心分类列举/创建/删除、connector 创建/查询及已有授权 OSS 文件导入。复用工作空间范围和单次请求策略；不会替调用方授权 bucket、上传云资源或自动轮询。
- 第一方 Anthropic Code Execution：当前官方工具声明、同步/流式容器 metadata、作用域绑定的容器续用，以及状态不明时禁止自动重提/切换。

本轮实现、专项回归及本地全量验收已完成。独立 ASR Realtime 及其余原始计划剩余项保持未完成。

第十一轮专项验收：Anthropic Code Execution 14 项及相关回放/OpenRouter回归、Qwen分类/connector/文件与两份矩阵共 87 项通过；独立 Qwen 实时 TTS 扩展回归 17 项通过。实时流额外修复 next 取消时终态结果丢失及 EOF 重复读取，覆盖可阻塞 close/send 的独立假传输。

本轮 Anthropic 实现另含有作用域的已上传文件 `container_upload` 输入，每次派发重新验证；同步与流式原生 usage 单独保留，并映射执行计数而不估算容器运行时长费用。Qwen 实时 TTS 支持当前已确认的 12 个模型 ID，按模型/地区区分音色来源和参数，不推断独立 ASR 或音色创建生命周期。

P1 后续核实项：当前 Anthropic Files GA 文档支持可选 `expires_in_seconds`/`expires_at`，本仓 `ProviderFileRef` 已保留到期元数据，但 `model_reference()` 转为 `ProviderFileSource` 时未携带该信息。现有模型输入及本轮 typed container uploads 只做来源作用域预检；第十二轮已补齐已知到期信息的投影与发送前校验；处理就绪状态仍需独立补齐，不能用 provider 服务端拒绝代替本地已知元数据检查。来源：[Files API 生命周期](https://platform.claude.com/docs/en/build-with-claude/files)。

### 第十一轮验证

- 全特性：1,448 项通过，0 失败，6 项示意文档代码忽略。
- 默认特性：1,427 项通过，0 失败，6 项示意文档代码忽略。
- 独立下游 crate：193 项文档测试及 1 项单元测试通过，56 项示意代码忽略。
- 全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式及 diff 检查通过。
- 离线打包及包内编译验证通过：569 个文件，19.8 MiB，压缩后 10.6 MiB。
- 中国提供方矩阵：594 条服务记录，72 个引用来源。Qwen实时TTS只更新已证实的地区服务记录，未提升模型能力单元格；Anthropic代码执行来源更新至当前版本与容器契约。

验收日志位于 `/private/tmp/llm-client-{allfeatures,default,rustdoc,downstream,package}-20260926k.log`；最终严格 Clippy 日志为 `/private/tmp/llm-client-clippy-final2-20260926k.log`。本轮新增 49 项回归，未执行真实账户/付费调用，未发布或提交 Git，未修改快照/缓存重构。原始 P0–P7 未全完成；下一轮优先处理文件到期元数据在模型输入中的保留/预检、Qwen独立ASR Realtime及其余已有证据的托管工具/知识库操作。

## 2026-09-26 第十二轮：文件到期预检、Qwen 实时 ASR 与托管检索

正在并行实现：

- 修复 P1 文件引用投影的到期元数据丢失，统一在模型输入与 typed 容器文件发送前校验已知到期状态；不把未提供的信息推断为已验证可用，也不重做快照/缓存架构。
- Qwen 独立实时 ASR：依据原生 ASR 握手、音频输入、VAD/手动提交、转写事件与 session.finish 契约实现，区别于 Omni 和异步文件转写。不从其他 WebSocket 协议猜测 clear/cancel 等事件。
- Qwen [Knowledge Search](https://help.aliyun.com/en/model-studio/knowledgesearch)：查询已发布的托管检索服务，保留 agent_id/version、图文 query 与按知识库过滤的独立协议；不自动创建/部署 Agent，不在客户端实现宿主 RAG 编排。

本轮实现、专项测试与统一验收已完成；按用户要求本轮结束后暂停目标，不继续其他剩余项。

P1 readiness 已完成：`ProviderFileRef::model_reference()` 与 `ProviderFileSource` 保留 provider 原始处理状态；模型请求前拒绝已知未就绪/失败状态。Gemini poll/resume 与 Qwen poll 返回的最新状态覆盖初始上传状态；Qwen 自动附件上传将 `processed` 引用用于模型请求。`get()` 在状态字段省略时保留旧状态，显式 `null` 则恢复未知。缺失或未知状态不被推断为已就绪，也不触发隐式刷新或重试。状态语义依据 [Gemini Files API](https://ai.google.dev/api/files) 与 [Qwen OpenAI-compatible File API](https://help.aliyun.com/en/model-studio/openai-file-interface)。

本轮另外补齐 Qwen [Knowledge Chat](https://help.aliyun.com/en/model-studio/knowledgechat) 的已发布托管问答 SSE：显式服务引用、调用方完整历史、作用域会话文件、原生阶段/工具结果和终态。客户端不执行这些托管工具，不持久化或裁剪历史，也不替调用方创建服务。

### 第十二轮验证

- 全特性：1,493 项通过，0 失败，6 项示意文档代码忽略。
- 默认特性：1,473 项通过，0 失败，6 项示意文档代码忽略。
- 独立下游 crate：199 项文档测试及 1 项单元测试通过，56 项示意代码忽略。
- 全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式及 diff 检查通过。
- 离线打包及包内编译验证通过：582 个文件，20.0 MiB，压缩后 10.6 MiB。
- 中国提供方矩阵：600 条服务记录，79 个引用来源；模型能力单元格未因新增服务而提升。

验收日志位于 `/private/tmp/llm-client-{allfeatures,default,downstream,package}-20260926l.log`；严格 Clippy 和 Rustdoc 最终日志分别为 `/private/tmp/llm-client-clippy-final2-20260926l.log`、`/private/tmp/llm-client-rustdoc-final-20260926l.log`。本轮未执行真实账户或付费调用，未发布或提交 Git，未修改快照/缓存算法。原始 P0–P7 尚未全部完成，文件处理就绪状态升级链等剩余任务保留，按用户要求在本轮收尾后暂停。

## 2026-09-26 第十三轮：处理就绪状态、模型参数契约与工具扩展

用户已明确恢复目标。本轮下列实现、专项测试与完整回归已完成：

- P1：已知文件处理状态从 upload/get/poll/resume 一致升级到模型输入，避免成功轮询后仍持有最初的处理中状态。
- P2：依据第一方文档核对 Gemini、DeepSeek、GLM、Kimi 的结构化输出组合，补齐有明确证据的缺失校验。
- P4：核对 Qwen、GLM Embedding 的模型维度和批量限制；不能把第一方限制套用到所有兼容网关。
- 全量交付：CI 增加包含 `realtime-websocket` 的全特性验证，保留默认依赖隔离与独立 tokenizer 编译。

逐操作反查又确认了三个已有文档证据、但公共接口尚缺的具体项目，已纳入本轮实现：OpenAI Responses 原生 Tool Search 与延迟加载；Responses 请求级缓存 key、模式、最小 TTL 和显式内容断点；Anthropic Programmatic Tool Calling 的 `allowed_callers`、调用者 metadata 与组合限制。它们分别属于原始 P3、P2、P3，不引入客户端内工具执行或历史管理。远端请求缓存参数与另一个任务完成的本地快照/缓存架构分开。

本轮还会复核原始剩余清单，区分缺少实现、官方契约未公开和真实账户未验收。当前矩阵共有 7,883 个 Unknown 模型能力单元格及 476 条 Unknown 服务记录；这些是证据缺口数量，不是待编码任务数量。没有证据时不会推断支持或不支持，也不会把本地测试作为线上验收。

### 第十三轮验证

- 最终全特性：1,523 项通过，0 失败，6 项示意文档代码忽略。
- 最终默认特性：1,502 项通过，0 失败，6 项示意文档代码忽略。
- 独立下游 crate：207 项文档测试及 1 项单元测试通过，56 项示意代码忽略。
- 全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式及 diff 检查通过。
- CI YAML 的 actionlint 和解析检查通过；未触发或声称通过远端 GitHub Actions。
- 离线打包及包内编译验证通过：595 个文件，20.2 MiB，压缩后 10.7 MiB；日志为 `/private/tmp/llm-client-package-20260926m.log`。

收尾修复了两项跨接口边界：普通 `caller` metadata 可在兼容 Anthropic Messages 的网关与云端包装协议中回放，而第一方程序化调用的声明与执行限制单独校验；OpenAI 的最小缓存 TTL 与最大 retention 按官方契约独立编码，允许已确认支持模型使用两者及显式断点组合。没有自动回退或缓存命中保证。

最终测试日志为 `/private/tmp/llm-client-allfeatures-final2-20260926m.log`、`/private/tmp/llm-client-default-final-20260926m.log`；严格 Clippy、Rustdoc、下游文档日志分别为 `/private/tmp/llm-client-clippy-final2-20260926m.log`、`/private/tmp/llm-client-rustdoc-final-20260926m.log`、`/private/tmp/llm-client-downstream-final-20260926m.log`。早期失败和修复前的成功日志保留，不替代这些最终结果。

按用户要求，每轮完成后先压缩上下文再进入下一轮。当前没有可调用的手动 compact 工具；[接续记录](implementation-handoff.md) 保留本轮结果和下一轮边界，不能将写入记录表述为已经执行 compact。原始 P0–P7 尚未全部完成，真实账户与区域验收仍未进行。本轮没有提交、推送、发布或付费调用，也没有重做本地快照/缓存算法。

## 2026-09-26 第十四轮：集合更新、Skills 请求与实时预检

第十三轮收尾并压缩上下文后重新核对当前代码。本轮确认并处理以下具体缺口，仍保留原始 P0–P7 范围：

- P7：`RealtimeSession::connect` 原先在建立连接后才生成、逐帧校验初始化配置；现改为先生成并校验全部初始帧，再调用传输层。无效配置或第二帧超限均不得握手或发送第一帧。实时核心 11 项、包含各提供方适配器及 WebSocket 后端的定向测试合计 43 项通过。
- P4：xAI 官方 [Collections API](https://docs.x.ai/developers/files/collections/api) 已文档化集合配置更新，但公共服务尚缺该入口；已补齐 typed 更新请求、Management API 路由、响应身份校验及单次发送测试。
- P3：Anthropic 官方 [Skills quickstart](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart) 已明确 `container.skills` 与 Code Execution 配合使用；已补齐预建及已上传 Skill 的请求声明、20 项上限、版本格式、自定义工作空间引用与容器复用校验。

### 明确保留的后续操作

- Anthropic Skills 独立资源与版本的上传、列举、查询、删除：官方 [Skills API](https://platform.claude.com/docs/en/api/http/skills) 有独立契约，不能由本轮容器请求支持代替。
- xAI Collections 的批量文档 metadata 查询、直接 Management API multipart 上传：当前双语服务说明明确未实现；下一轮逐项核对当前官方契约后补齐。全局文件删除已由 `FileService::delete` 的 xAI adapter 提供，不能仅因 Collections facade 没有同名方法就算作全库缺口；两种上传引用的接续与不同上传大小限制仍需审计。
- 原计划列出的真实账户、区域可用性与端到端恢复验收仍未进行；mock、公开文档和编译通过均不能证明这些项目完成。

当前阶段不是完整目标的完成声明；本轮最终集成验证另行记录。

后续 P1 已确认参数缺口：当前 [xAI Files upload reference](https://docs.x.ai/developers/rest-api-reference/files/upload) 明确 `POST /v1/files` 上限为 50 MB；`src/files/policy.rs` 的 `Adapter::Xai` 仍配置为 512 MiB，而 Collections 上传已经使用 50,000,000 字节。下一轮统一普通文件上传的上限及流式/缓冲无副作用回归；本轮集合更新与 Skills 的最终验证不代表该旧缺口已修复。初始化/分块上传的页面未给出请求参数，不能用它们推断普通 multipart 支持更大文件。

### 第十四轮验证

- 全特性：1,528 项通过，0 失败，6 项忽略；默认特性：1,507 项通过，0 失败，6 项忽略。
- 下游独立 crate：209 项文档测试及 1 项单元测试通过，56 项示意代码忽略。
- 全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式和 diff 检查通过。
- 离线打包与包内编译通过：595 个文件，20.2 MiB，压缩后 10.7 MiB。
- 最终日志：`/private/tmp/llm-client-{allfeatures,default,downstream,rustdoc,package}-20260926n.log` 与 `/private/tmp/llm-client-clippy-final-20260926n.log`。早期 `check` 与首次 `xai-collections` 日志记录共享源代码尚在编辑时发现的编译问题，不能作为最终结果。

本轮所有验证进程已退出，两个并行实现任务已完成。未调用真实提供方、未提交、推送或发布；原始目标保持未完成。接续优先级为上述 xAI 文件大小校验、Anthropic Skills 资源/版本管理、xAI Collections 已公开剩余操作。每轮结束后先按用户要求压缩上下文再接续；当前无手动 compact 工具，保存接续记录不代表执行了压缩。

## 2026-09-26 第十五轮：文件上限与原生资源操作

上一轮确认的具体缺口已实现，当前进行本轮统一验收：

- P1：普通 xAI multipart 文件上传与 Collections 上传统一为 50,000,000 字节。缓冲和流式超限均在发送前拒绝，流式输入不被读取；恰好等于上限可通过预检。文件服务专项 27 项通过。
- P3：独立 Anthropic Skills 与版本资源生命周期，以及显式版本内容下载。资源 CRUD 使用当前 GA 契约；内容下载参考仍位于官方 Beta 文档，不能将其一起宣称为 GA。返回 ZIP 原始数据，不自动解压或执行。
- P4：核对并补齐 xAI Collections 批量文档 metadata 和直接 Management API multipart 上传；继续检查与公共文件服务的引用接续。
- P0：西方提供方证据矩阵增加 9 条 Skills 和 2 条 xAI Collections 独立服务记录、3 个官方来源，当前合计 56 条服务记录与 53 个来源；模型能力单元格没有因服务端点存在而被提升。真实账户验证仍为 `not_run`。

本轮最终集成验证另行记录；上一轮的 xAI 文件上限待办现已修复，原始全量目标尚未完成。

### 下一轮已确认的 P3 缺口

Anthropic 第一方 Messages Remote MCP 尚无类型化请求入口：当前 encoder 不编码 `mcp_servers`/`mcp_toolset`，已有 `RemoteMcpConfig` 与逐次授权仅覆盖 OpenAI/xAI Responses。依据 [MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector) 与 [Messages Beta API](https://platform.claude.com/docs/en/api/beta/messages/create)，下一轮应补齐 Anthropic 专属 server/toolset、逐请求 token、当前 beta 与原生 MCP 回放；不要套用 OpenAI 审批策略。`mcp-client-2026-09-15` 的工具列表回放及与 `inline-tools-2026-09-15` 的组合须按 [mid-conversation guide](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages) 核验。`mcp_tool_use` 分块 JSON 参数的组装需要独立回归，不能仅凭未知原生块保留宣称已支持完整流式回放。

### 第十五轮验证

- 全特性回归 1,555 项通过，0 失败，6 项忽略；该运行后补入的下载读取超时用例已在最终全特性 Skills 专项中验证，专项 14 项全部通过。没有生产代码在这两次验证之间变更。
- 最终默认特性 1,535 项通过，0 失败，6 项忽略，包含上述新增超时用例。
- 下游独立 crate：213 项文档测试及 1 项单元测试通过，58 项示意代码忽略。
- 全目标/全特性严格 Clippy、警告视为错误的 Rustdoc、格式与 diff 检查通过。
- 离线打包与包内编译验证通过：601 个文件，20.3 MiB，压缩后 10.7 MiB。
- 最终日志：`/private/tmp/llm-client-{allfeatures,default,downstream,rustdoc,package}-20260926o.log`、`/private/tmp/llm-client-skills-final-20260926o.log`、`/private/tmp/llm-client-clippy-final-20260926o.log`。早期编译失败日志和缺少尚未创建测试目标的日志保留，不替代最终结果。

本轮三个并行实现/审计任务与独立 MCP 审计均已完成，全部验证进程已退出。未请求真实提供方、未提交、推送或发布；没有修改快照/缓存架构。原始目标未完成，下一轮优先按上方已确认的 Anthropic Messages MCP 契约补齐请求、逐次授权和原生流式回放。接续记录已更新，不能把记录写入当作已经 compact。

## 2026-09-26 第十六轮：Anthropic Messages MCP

按上一轮已确认缺口并行实施原生请求与流式回放：独立 Anthropic server/toolset 类型、逐请求 Secret 注入、当前工具列表固定 beta、工具级 enabled/defer_loading，以及 MCP 原生块 JSON 参数组装。遵循当前官方 connector 契约：每个 HTTPS server 恰好对应一个 toolset，未知工具名配置不从本地列表推断为错误，不照搬 OpenAI 的审批字段。

矩阵的 Opus 5.5 示例证据不被解释为其他模型不支持；其他模型仍保留 Unknown。此次处理协议入口，真实账户、远端服务器授权与区域可用性单独验收。所有客户端工具执行、OAuth 获取/刷新和历史管理仍由宿主负责。

本轮核查另外确认中途 system 消息及工具增删是独立的未完成协议面：当前 `src/codecs/anthropic/encode.rs` 无条件拒绝 `MessageRole::System`；[当前官方指南](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages) 则明确列出支持模型，并将内联工具定义/MCP 工具集放入 system 消息的 `tool_addition` block。它需要独立的模型/位置约束和 `inline-tools-2026-09-15` 组合，不能仅添加 beta header 便宣称支持。本轮顶层 MCP server/toolset 与列表回放不替代该后续实现。

第十六轮验证：全特性 1,579 通过、默认特性 1,558 通过，两者均 0 失败、6 忽略；随后仅修正新增测试的两处 Clippy 风格问题，最终全特性 MCP 专项 11 项通过，生产代码未变。严格 Clippy、Rustdoc 和下游 217 文档测试 + 1 单元测试通过（58 文档测试忽略）。日志前缀 `/private/tmp/llm-client-`、后缀 `20260926p.log`；早期失败日志保留。用户要求本轮结束后暂停 goal，后续中途 system/inline tools 和原始范围其他未完成项等待恢复。

第十六轮离线打包与包内编译通过：605 文件、20.4 MiB（压缩后 10.7 MiB）；格式与 diff 检查通过。本轮结束，goal 按用户要求暂停。

## 2026-09-26 第十七轮：第一方 Anthropic 中途指令与工具变更

根据当前官方 mid-conversation-system-messages 指南，实施受支持 wire model 的 System 历史分组位置校验、原生 text/tool_addition/tool_removal、自定义工具定义和引用增删、MCP 配置内联位置、header 组合及逐消息工具状态。顶层服务器连接和逐请求凭证保持原有边界；不将工具集重复放入顶层 tools。

缓存校验纳入 definition 内部标记并按消息顺序计数，拒绝 block/definition 双重标记、与 typed breakpoint 重叠及 deferred 定义缓存。工具状态按可用定义校验计数/已知序列化字节上限，未知 MCP 动态列表及服务端渲染字节由服务端校验。

后续独立缺口：turn-scoped clear_at、逐消息 output_config.effort、内联 strict schema/程序化 caller、通用服务端工具内联定义、云平台路由。当前切片对未完成组合显式拒绝，不宣称这些能力已完成。宿主仍负责权限、工具执行、历史与刷新。本轮验证结果见下。

第十七轮验证（2026-09-27完成）：全特性1,600通过、默认特性1,579通过，均0失败、6忽略。全特性完整回归后补充整组MCP按定义重加时清除成员覆盖状态，默认完整回归和最终全特性32项专项包含此修正；最终严格Clippy、Rustdoc、223下游文档测试+1单元测试通过（58文档测试忽略）。离线打包及包内编译通过：609文件、20.5MiB（10.7MiB压缩）。格式和diff通过，日志20260926q，过程失败保留；没有真实账户/付费调用。goal继续active，原始完整目标未结束。

## 2026-09-27 第十八轮：Anthropic 消息级控制

增加类型化消息选项与wire clear_at/output_config.effort；支持单轮提醒与空内容effort-only位置例外，同时保留连续system区段约束。选择独立beta，不把控制伪装成content block，不改写已清除的历史。其他内置协议显式拒绝Anthropic消息选项，错误在附件读取之前被发现。缓存标记与单轮提醒互斥；当前请求effort报告仅在后续user出现后更新，顶层设置保持原值。公共ConversationMessage字面量需新增anthropic字段，构造器默认None；不存在旧API兼容分支。新增11项专项及协议serde回归，最终验证见下。

第十八轮最终验证：冻结源码全特性1,612通过、默认特性1,591通过，均0失败、6忽略；严格Clippy、Rustdoc、格式、diff通过；下游227文档测试+1单元测试通过（58文档测试忽略）；离线打包及包内编译通过，610文件、20.5MiB（10.7MiB压缩）。日志20260927r，最终allfeatures-final/clippy-final2；早期构建期间的预检修正后已完整重跑，没有靠旧构建结果支持完成声明。无真实模型/付费调用，goal active。

下一项已确认P3缺口：独立Anthropic Web Fetch类型/请求适配（不是Qwen）、Browser/Computer客户端工具集的声明与toolset_name往返。依据Anthropic官方tool-reference、web-fetch-tool与browser-use-tool指南；原始其他provider/域的范围不因此缩小。


## 2026-09-27 第十九轮：Anthropic 第一方 Web Fetch

新增独立类型化 Web Fetch 声明和四种当前版本，明确限制 use_cache/response_inclusion 的适用版本，保留 max_uses/max_content_tokens 的零值及省略语义。allowed_callers、strict、defer_loading、缓存标记与 URL 来源过滤均有类型和本地校验；Direct-only 不错误套用动态过滤模型限制。路径过滤语法原样保留，但文档明确 Web Fetch 不匹配含路径的条目。

请求在附件读取前验证，typed/native Fetch 都禁止结果不确定时自动重试或切换。Fetch 缓存标记按实际 MCP 后、system 前的编码位置参加四断点及 TTL 顺序检查；strict 与引用组合共享现有结构化输出限制。原生结果、HTTP 200 内工具错误和流式输入保留为 ProviderContent，web_fetch_requests 作为可选用量保留零与缺失的区别，不推算费用。新增19项专项与双语下游示例。

冻结后全特性1,631、默认特性1,610通过，均0失败、6忽略；严格Clippy、Rustdoc通过；下游231文档测试+1单元测试通过，58文档测试忽略。离线打包及包内编译通过：615文件、20.6MiB（10.8MiB压缩）；格式与diff通过。日志后缀20260927s。独立只读审查未发现已确认的第一方阻断缺陷。无真实提供方调用。

接续：Browser/Computer稳定客户端工具集与toolset_name完整往返；工具执行仍归宿主，声明不得混淆HostedTool的提供方执行语义。Foundry Web Fetch须独立区分托管类型与支持版本。内联strict/PTC、通用服务端内联定义及原始P0–P7其余范围保持待办，当前全目标仍active。


## 2026-09-27 第二十轮：Anthropic Browser / Computer 客户端工具集

增加独立ChatRequest.anthropic_client_toolsets声明，支持稳定20260801的Browser/Computer成员配置、默认启用状态、direct-only caller、工具搜索/延迟加载、缓存、模型/协议与命名冲突限制。宿主仍执行工具，不纳入HostedTool。ToolUse、ToolResult和ToolCallDelta完整保留toolset_name；tool_uses()改返回(id,namespace,name,input)，相关Rust字面量及双语API示例同步迁移，不保留旧API兼容分支。

历史校验覆盖已知调用/结果命名空间、允许的原生结果块、标签管理结果和browser_state结构、可空状态/下载字段。截图与其他成员的图片/文本输出约定按官方说明由宿主执行，不误当作API硬限制。独立审查后补齐内联custom定义命名冲突、下载大小非负整数、MCP与客户端工具集合计上限的附件前预检。缓存计数/TTL顺序与实际wire顺序一致；本地token估算显式列出未计入的提供方成员定义。

最终源码冻结后，18项工具集专项通过；全特性1,651通过、默认特性1,630通过，均0失败、6忽略；严格Clippy、警告为错误Rustdoc、235下游文档测试+1单元测试通过（58文档测试忽略）。离线打包及包内编译通过：622文件、20.7MiB（10.8MiB压缩）。格式/diff通过；615个验证输入哈希无变化。最终pipeline session36015 exit0，无运行中的Cargo。日志/private/tmp/llm-client-{toolsets-final2,clippy-reviewed,allfeatures-final,default-final,rustdoc-final,downstream-final,package-final}-20260927t.log；早期编译、fixture与审查前验证日志保留，不替代最终证据。无真实提供方调用、提交或发布。

下一轮必须补充Browser/Computer独立操作的P0证据矩阵；继续Vertex Tool Search模型限制和中途控制（仅工具引用，不包含API-only内联定义）。Foundry共享endpoint且部署可自定义名称，需要typed hosting/underlying-model身份后才能启用托管类型专属能力。Foundry Web Fetch和GoogleCloud稳定工具集的不同官方入口存在冲突，保持显式拒绝并核实，不把指南或mock成功当成全平台上线验收。完整原始P0–P7范围保持不变，goal active。


## 2026-09-27 第二十一轮：Vertex Claude 与工具集证据矩阵

Vertex Claude 接入 Tool Search 的精确模型 ID 校验、中途 System、逐消息 effort/clear_at 和引用型工具增删；云平台使用独立引用变更 beta，继续拒绝 Claude API 专属的内联工具定义/MCP。稳定 Browser/Computer 工具集与命名空间历史亦已接入。Vertex 编码与预检共享 Anthropic 校验，错误在附件读取前发现；普通 inference 控制也统一提前校验。新增双语 Vertex 指南和下游编译示例。

依据本轮当前官方 Vertex 指南，稳定 Browser/Computer 已被明确列为支持，第二十轮的旧冲突结论被替代。P0 增加两个独立操作和对应云平台服务证据：7,878 个模型单元格（1,412 supported / 702 unsupported / 5,764 unknown）、60 个服务记录、57 个官方来源。受支持模型未出现在现有目录时不捏造模型行；Foundry 稳定工具集仍按官方明确排除记录为 unsupported。

首轮完整测试发现旧 OpenRouter 回归中的第一方 Anthropic 对照夹具仍使用 any/model，已改为受支持的 claude-opus-4-6；OpenRouter 自身的模型透传未更改。修正后的该专项 11 项通过。随后完整回归定位到通用 inference 早期校验误用整个 profile 的首个同 wire ID 模型行，已改为使用 CodecContext::for_model 保留的选中模型行；既有回归覆盖正反目录顺序及 complete/stream。完整门禁已在最终源码冻结后重新执行，结果见下。

下一轮具体契约已确认：Foundry 需在每个 ModelProfile 上增加 typed hosting（Azure/Anthropic）及 underlying model ID；request_model 继续传自定义部署名，不从 URL 或 billing_model 推断。Tool Search 按托管类型可用模型与工具官方列表的交集验证。Web Fetch 在 Azure 托管仅支持 20250910，Anthropic 托管支持各版本并保留动态过滤模型限制。MCP 在 Foundry 使用 2025-11-20 beta；2026-09-15 的 listing/pinned tools 为 Claude API 专属，即便 Anthropic 托管的 Foundry 也不能推断支持。当前官方 feature overview 的 Tool Search/MCP 不带托管类型限制脚注，旧相反结论作废。来源见 [Foundry 指南](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)、[Tool Search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)、[Web Fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)、[MCP](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)。

第二十一轮最终验证：全特性 1,666 通过、默认特性 1,644 通过，均 0 失败、6 忽略；严格全目标/全特性 Clippy、警告为错误 Rustdoc、格式和 diff 检查通过。下游 239 项文档测试 + 1 项单元测试通过，58 项文档测试忽略。离线打包及包内编译通过：628 文件、19.9 MiB（10.8 MiB 压缩）。621 个冻结输入哈希和文件集合完全一致。最终 pipeline 12811 exit0，无运行中 Cargo；日志 /private/tmp/llm-client-{clippy,allfeatures,default,rustdoc,downstream,package}-final2-20260927u.log。早期两次完整运行的失败日志保留，已由最终完整重跑覆盖。全部 worker 完成，无真实提供方或付费调用，无提交、推送或发布。goal 保持 active，原始 P0–P7 尚未全部完成，下一轮按上述 Foundry typed identity 与协议边界推进。接续记录已保存，当前没有手动 compact 工具，不将保存记录表述为已经压缩。


## 2026-09-27 第二十二轮：Foundry 部署身份及工具适配

新增每模型 FoundryDeployment（hosting + model_id）与 ModelField::Foundry 覆盖/清除/继承；request_model 继续发送自定义部署名，billing_model 仅定价。CodecContext 选中模型行保留身份，wire-ID-only 上下文中的冲突身份显式拒绝。托管与模型依官方精确表验证，不从端点、部署别名或价格推断。

Tool Search 按 Foundry 托管模型与工具支持列表的交集启用；Web Fetch 的 Azure 托管仅基础 20250910 与直接调用，Anthropic 托管支持四版本且动态过滤另受底层模型限制。Foundry MCP 使用 2025-11-20，支持逐请求凭证和原生 use/result 回放；Claude API 专属的 2026 pinned/listing/inline 请求及原始注入在本地拒绝。MCP 不要求没有官方依据的模型白名单，也不把 hosting 身份作为基本连接器前提。

高层预检与编码保持一致，凭证仅注入配置中的 HTTPS Messages 端点；远端执行结果未知时不自动重试/切换。跨模块审查补齐 Azure Web Fetch 调用者限制及 Foundry/Vertex 提供方提示缓存的共享四标记/TTL 校验；未重做本地快照或缓存架构。

P0 增加五个服务证据（Foundry Tool Search、分托管 Web Fetch、Foundry MCP 2025、Claude API MCP 2026 pin/listing）：65 服务、58 来源、111 操作，模型矩阵仍为 408 模型和 7,878 单元格。新版 MCP 在 Foundry 的官方支持未建立，客户端拒绝不等同线上实测不支持。双语 Foundry 指南、已有工具指南、API 文档和下游编译示例已同步。

本轮用户要求结束后暂停：完整验证已完成，接续记录已保存；本轮结束后暂停 goal，不自动进入下一轮。

第二十二轮最终验证：107 项统一专项通过；冻结输入上全特性 1,691 通过、默认特性 1,670 通过，均 0 失败、6 忽略。严格全目标/全特性 Clippy、警告为错误 Rustdoc、格式和 diff 检查通过；下游 243 文档测试 + 1 单元通过，58 文档测试忽略。离线打包与包内编译通过：635 文件、20.0 MiB（10.8 MiB 压缩）。628 个冻结输入的哈希与文件集合均无漂移。最终 pipeline 13449 exit0；无运行中 Cargo，全部并行任务已结束。日志 /private/tmp/llm-client-{allfeatures,default,rustdoc,downstream,package}-20260927v.log、clippy-final3-20260927v.log。过程中旧共享门槛、测试夹具和 Clippy 风格失败日志保留，已由最终完整门禁覆盖。无真实账户/付费调用，无 Git 提交、推送或发布。原始 P0–P7 仍未全部完成，按用户要求在本轮结束后暂停后续推进。

## 第二十三轮：并行补齐已确认的剩余协议（本地验收完成）

用户恢复实施并指定 GPT-6 Luna max 并行工作。本轮围绕 Anthropic 内联 strict/PTC、Foundry Anthropic 托管执行与作用域、xAI Collections 检索模式、OpenAI Embedding 限制及目录、OpenAI Realtime 工具往返展开；仍保持原始 P0–P7 范围。Qwen 知识库更新经代码复核已存在，不重复计作新增。本轮新增功能已接入；最终统一验收结果在本节末记录。


本轮逐操作核查形成以下有限实施清单，后续以该清单和验证缺陷收敛，不将 Unknown 单元格逐个计为编码任务：

- Anthropic 内联 strict/PTC、typed Code Execution/Web Fetch 放置与时间顺序校验。
- Foundry Anthropic 托管执行/容器、Files 和 custom Skills 资源生命周期及精确作用域；Azure 和版本内容下载的明确排除。
- OpenAI Embeddings 模型限制/独立模型目录；Realtime 工具声明/批量结果/显式继续/图像/音色；Background 显式删除。
- xAI Collections 检索模式；xAI JSONL 文件式 Batch；Qwen Batch 错误文件与查询身份；OpenRouter Batch 日期/模态/endpoint slug。
- Qwen 独立 Audio Generation、Audio 3.x 文件 ASR、LiveTranslate 独立协议；MiniMax ASR SSE 与 HTTP TTS 字幕控制；OpenRouter TTS 参考音频。
- 统一测试、文档示例、静态检查、离线打包和能力证据记录。真实账户、付费调用、未公开或冲突契约单列，不能用 mock 成功消除。


最终契约复核纠正了三项遗漏，继续纳入本轮实现：GLM 当前 Knowledge `upload_document/{id}` 文件上传（与已废弃 Agent 上传接口分开）；MiniMax HTTP TTS SSE（由官方 CLI 确认 framing/hex/status2，未知字幕结构只保留原生字段）；OpenAI GPT-Live 主 WebSocket（独立 `/v1/live/sessions`，不与旧 `/v1/realtime` 事件混用）。OpenAI Vector Store 搜索经官方 OpenAPI 与 SDK 复核，request 不包含 next_page/cursor，响应 token 不能反推可发送的分页参数；该项不是已确认编码缺口。


GLM Realtime 另已按官方 SDK 补齐函数工具声明、精确 call_id 结果回传与显式继续，工具续接不受音频 VAD 模式限制。新增协议只做 mock/本地回环验证；未执行真实账户或付费调用。


第二十三轮最终结果：本节有限清单中的已确认编码任务全部完成。使用五个 GPT-6 Luna max 并行工作任务，并对 Anthropic 内联工具、Foundry、GLM 上传/Realtime、GPT-Live 等边界进行独立复核。GPT-Live 两处审查问题（缺失模型身份、终止事件后的排队发送）已修复并加入回归。未重做其他 session 的本地快照/缓存架构。

验证：最后八项统一专项 87 通过；全特性测试 1,809 通过、默认特性 1,788 通过，均 0 失败、6 忽略。严格全目标/全特性 Clippy、警告为错误 Rustdoc 通过。下游独立 crate 的 267 项文档测试及 1 项单元测试通过，56 项历史示意测试忽略。初次下游检查暴露 Qwen 示例外部 Bytes 依赖和 MiniMax 示例缺少异步上下文，已仅修改八个服务指南，补齐 GLM/音频双语示例；最终下游检查全部通过，生产源码与全量测试时一致。

离线打包及包内编译通过：654 文件，20.7 MiB，压缩后 10.9 MiB。格式与 diff 检查通过。653 个冻结输入（不含本计划和接续记录）的哈希与文件集合无漂移；最终文档示例修正后重新冻结并验证。日志位于 `/private/tmp/llm-client-`：`allfeatures-final-20260927w.log`、`default-final-20260927w.log`、`rustdoc-final-20260927w.log`、`clippy-final3-20260927w.log`、`downstream-final2-20260927w.log`、`package-final-20260927w.log`，更新最终记录后的包验证另存 `package-final2-20260927w.log`。此前编译、夹具、Clippy 和下游失败日志保留，不当作通过记录。

证据矩阵当前为：中国提供方 606 服务/86 来源/95 操作，其他提供方 101/61/147，OpenAI-Gemini 37/96/73；模型行与单元格未因本轮服务操作而虚增。全部是文档证据，不代表真实账户调用通过。

剩余事项分为两类：真实账户、区域/权限、服务端恢复和设备端到端验收尚未执行；OpenAI Speech SSE 文本对齐时间戳、Z.AI 国际 TTS 路由、Anthropic 内联 Tool Search/deferred catalog 组合仍缺明确契约。OpenAI Vector Store 搜索响应 next_page 不对应当前已公开的请求分页参数，不列作已确认编码任务。Unknown 能力单元格不逐个计作剩余任务。原始 P0–P7 不据本地测试标记为线上全部完成。本轮无真实提供方/付费调用，无提交、推送或发布。

## 第二十四轮：剩余契约复核与只读验收入口

沿用 GPT-6 Luna max 并行工作，针对上一轮保留的三项契约做有限复核，不将所有 Unknown 能力单元格重新展开为实现任务。

- Anthropic 已确认的组合已由现有实现覆盖：顶层 Tool Search、顶层完整 deferred tools 目录以及 system `tool_addition` 的 `tool_reference`。补充一项组合回归及双语工具搜索/会话指南。消息内按值新增定义是否被托管搜索索引仍未由官方说明建立，不将它等同于已支持的引用路径。
- OpenAI Speech API/TTS guide 未建立 SSE 文本/音频对齐字段；Z.AI 国际 OpenAPI/index 未建立 TTS 路由。Z.AI Java SDK 的 TTS 概述不足以确认国际路由与 schema。相关指南精确使用“契约尚未建立”，不声称厂商明确不支持。
- 复核 xAI 音色目录时确认 `GET /v1/tts/voices/{voice_id}` 未接入，现新增 `XaiAudioService::get_voice` 与 `XaiVoiceDetails`。详情保留 scope/native/request ID，返回 ID 必须精确匹配，路径分隔符、转义与 dot segments 在发送前拒绝，GET 只发送一次。增加四项回归；独立复核无确认缺陷。
- 新增 `examples/xai_voice_acceptance.rs` 与[真实验收说明](provider-acceptance.md)：默认只输出计划，不读密钥或联网；显式执行时，使用环境凭证查询内置音色目录与首个音色详情，最多两个只读 GET，不调用生成、不创建资源、不自动重试、不输出凭证或原生响应体。空目录/失败不能算整体验收通过。
- West 证据矩阵现为 102 服务/62 来源/148 操作；模型行与单元格不变，全部真实调用状态仍未验证。

当前实际网络状态：一次受限环境的目录检查在 transport 层失败，未获得提供方响应。外部网络升级被自动审批拒绝，明确理由是当前任务没有授权向 xAI 发送环境凭证；没有绕过拒绝或继续真实请求。两个只读 GET 的真实验收等待用户明确授权，不能记为通过。其他提供方需要相应账户/区域/权限与具体操作范围。此次没有生成或资源写入，也没有提交、推送、发布。

本轮七个相关全特性专项目标共 60 项通过。严格全目标/全特性 Clippy 与默认只读入口预览通过。后续文档与打包验证结果在本节末记录。本轮仅新增一个独立只读操作，不将上一轮 1,809/1,788 全套测试数字冒充本轮重新执行的结果。

第二十四轮本地验收完成：60 项专项通过；严格 Clippy、Rustdoc、默认不联网预览通过；下游 267 文档测试及 1 单元测试通过（56 忽略）；离线打包与包内编译通过（657 文件，20.8 MiB / 10.9 MiB 压缩）。fmt/diff 检查及 656 个冻结输入哈希/文件集合校验通过。日志均为 `/private/tmp/llm-client-` 下的 `focused-20260927x.log`、`clippy-20260927x.log`、`xai-acceptance-dry-final-20260927x.log`、`downstream-20260927x.log`、`rustdoc-20260927x.log`、`package-20260927x.log`。真实目录请求的 transport 失败单列 `xai-acceptance-live1-20260927x.json`；自动审批拒绝后的执行没有启动。没有运行中 Cargo 或工作子任务，下一步仅在明确授权后运行已经准备好的真实只读验收。


## 第二十五轮：功能优先收敛（2026-09-27，功能整合完成）

按用户最新要求，先补齐功能，统一测试留到功能收敛后执行。本轮不调用真实提供方接口，不重复执行全套测试；上一轮通过数量不能作为本轮改动的验证结果。快照/缓存架构保持其他 session 的实现。

已落地、待统一验证：

- OpenAI/Qwen Code Interpreter 请求固定到首选连接，关闭自动连接切换与失效附件修复重发，避免容器代码重复执行。
- Gemini Live JPEG/PNG 视频帧输入，映射到 `realtimeInput.video`；批量工具结果以一个 `toolResponse.functionResponses` 消息提交，预检调用 ID、重复项及 JSON 对象。同步修正工具调用名称表新增条目的字节计数。

并行实施：xAI Voice 会话参数与输入清理、OpenAI Realtime/Live 明确缺失命令、Code Interpreter 已有容器与自动挂载文件。资源操作与其他协议继续以官方契约对照实际入口，文档未建立的能力不猜测实现。

统一验证待办：Gemini 视频帧 MIME/空输入/帧大小、批量结果未知或取消 ID/重复 ID/非法对象/无部分发送；工具名称存储边界；Code Interpreter fallback 与失效文件响应均仅发送一次；新增容器和实时控制接口的编译、协议 fixture 与下游示例，随后执行全套交付门禁。

第二十五轮继续确认并实施：Gemini Chat 类型化音频输入（GenerateContent/Vertex，共用有界 inlineData 编码）；OpenAI、Qwen、GLM 与 xAI Realtime 清空未提交输入；GPT-Live 启动历史及显式存储选项；Gemini 双说话人 TTS 与 Beta Voices 资源。并行任务完成后记录精确入口，现阶段不标为测试通过。


新增范围与证据：

- [Gemini Chat 音频输入](gemini-chat-audio.md)：类型化用户音频、标准 Base64 与 MIME 映射、完整请求体限额；不是 OpenRouter 输出音频配置的透传。
- [Gemini Live](gemini-live.md)：视频帧与批量函数结果。
- [OpenAI Realtime](realtime.md)：ClearAudio、RetrieveItem、DeleteItem、TruncateAudio；[GPT-Live](openai-live.md)：类型化启动历史与默认关闭的 store 选项。
- [Qwen Omni](qwen-realtime.md) / [GLM Realtime](glm-realtime.md)：仅补已明确的输入音频清理，不从其他协议推断 session.finish。
- [Gemini Speech](gemini-speech.md)：两名说话人、逐段说话人/风格与单次/SSE 入口共用编码。
- [Gemini Voices](gemini-voices.md)：Beta 资源 create/list/get/delete；操作证据已加入矩阵，OpenAI/Gemini 矩阵变为 41 个服务记录、97 个来源、77 个操作定义，在线状态保持 not_run。

Vertex TTS 是独立的提供方路由，继续核对/实现；旧版第一方 GenerateContent TTS 不因兼容旧版而重复增加。


当前编译记录：第一次 `cargo check --locked --offline --all-targets --all-features` 未通过，发现 Gemini Voices URL query serializer 参数类型与 xAI connect 参数可变性两类错误；已交回对应实现修复。该检查不是测试运行，最终编译结果待各并行功能整合后更新。日志：`/private/tmp/llm-client-round25-check.log`。


第二次集成编译通过：`cargo check --locked --offline --all-targets --all-features`（日志 `round25-check2.log`）。该结果发生在 Vertex TTS 注册和 Gemini 工具调用原子登记修正之前，最终整合仍需再编译。尚未运行测试。

Gemini Live 长会话修正：只有工具结果成功进入有界发送队列后才释放调用名称记录，队列满/关闭/帧失败保留状态；服务端批量工具调用先完整校验并暂存，全部有效后再原子写入名称表，避免半批失败留下宿主从未收到的调用。新增验证待办包含这些路径和 1024 次以上连续工具调用。


### 第二十五轮功能整合清单

1. OpenAI Code Interpreter：作用域已有容器、自动容器 Files 挂载、账户/端点/就绪/过期与重复 ID 预检；Qwen 拒绝 OpenAI 专有参数；执行请求禁自动重发。
2. Gemini Chat：音频内容块、标准 Base64 与格式编码、混合附件上传前完整内联大小预检、最终体积校验。
3. Gemini Live：JPEG/PNG 帧、批量函数结果、批量调用原子登记、成功入队后的名称记录回收。
4. OpenAI Realtime：清理音频缓冲、读取/删除会话条目、截断助理音频；GPT-Live：类型化启动历史与显式 store。
5. xAI Realtime：类型化会话控制、JSON/二进制及 Opus 音频、脚本消息/单次响应指令、原生生命周期映射、账户/端点/模型绑定恢复引用。
6. Qwen Omni / GLM Realtime：清理未提交音频缓冲。
7. Gemini Speech：两名说话人及逐段说话人/风格配置，单次与 SSE 合成共用入口。
8. Gemini Voices Beta：提示词/参考录音创建、过滤分页、查询、删除，存储 ID/无存储 key 互斥校验与作用域安全合成辅助入口。
9. Vertex Gemini-TTS：独立 project/location/account/model 路由、调用方 bearer token、单人/双人合成及 SSE、PCM 格式检查、显式终态/提前 EOF 与中断字节数。

清单中的功能代码已整合，仍待统一测试；新增 fixture 不能写成已通过。全部现有文档矩阵在线验收继续保持 not_run。当前尚无明确请求契约的项目：OpenAI Speech SSE 文本对齐、Z.AI 国际 TTS、Anthropic 仅消息内按值定义的 Tool Search 可发现性、xAI 会话条目删除/截断的完整请求字段。它们与“实现了但未测试”分别记录。

统一测试入口包括新增 Vertex/Gemini Voices 资源、各音频/实时协议、Code Interpreter 副作用边界、默认/全特性测试、严格 Clippy、Rustdoc、下游示例和打包。用户要求先完成功能，本轮仅执行编译与格式整合检查，完整测试尚未运行。

最终整合编译通过：`cargo check --locked --offline --all-targets --all-features`，日志 `/private/tmp/llm-client-round25-final-check2.log`，耗时 29.91 秒，无诊断。初次编译错误均已修正。完整测试、Clippy、Rustdoc、下游示例、打包及真实账户验收仍待统一阶段，不能以编译成功替代。

最终 `cargo fmt --all -- --check` 与 `git diff --check` 通过。


## 第二十六轮：统一验收与收尾（2026-09-27，已完成）

第二十五轮九项功能已完成本地统一验收。沿用用户指定的 GPT-6 Luna max 并行补充各 provider 的回归，由主任务统一运行 Cargo；未修改其他 session 的快照/缓存架构，未调用真实提供方、提交、推送或发布。

本轮补齐并验证的关键路径：

- Gemini Chat 音频格式/Base64/角色、完整请求体上限，以及混合附件上传前拒绝超限和合法请求的集成路径。
- Gemini Live JPEG 输入、批量工具调用/结果的原子性、队列满时保留重试状态、成功入队后清理、1,025 次连续调用；OpenAI Realtime/Live 历史与控制事件，xAI 二进制音频/恢复作用域，Qwen/GLM 清理缓冲。
- Code Interpreter 在完整客户端的普通/流式网络错误和 HTTP 500 后不切换重发；自动文件上传遇到失效引用时，普通请求触发修复，而代码执行请求仅发送一次。
- Gemini 双说话人合成、Voices 创建/分页/作用域/ID-key 互斥/未知提交结果/删除与脱敏；Vertex TTS 请求路由、bearer、双人合成、PCM MIME、SSE 终态、阻断及提前 EOF。

实际修复：`GeminiVoicesError` 的 Debug 不再输出原始提供方响应体，避免异常创建响应中的音色 key 被打印。另修正测试夹具的模型限制假设、提前注入后续合法事件的竞态，以及 Gemini 图像能力的旧断言；缺失文件修复对照改用真实自动上传分支。严格 Clippy 提出的布尔表达式和 Option 检查已调整。Gemini Voices 中文指南已补齐翻译。

最终验证（日志目录 `/private/tmp`，共同前缀 `llm-client-round26-`）：

| 验证 | 结果 | 日志后缀 |
| --- | --- | --- |
| 默认特性测试 | 1,836 通过，0 失败，8 忽略 | `default.log` |
| 全特性测试 | 1,858 通过，0 失败，8 忽略 | `all-final.log` |
| 下游独立 crate | 273 文档测试 + 1 单元测试通过，0 失败，62 文档示例忽略 | `downstream.log` |
| 严格 Clippy，全目标默认/全特性 | 通过 | `clippy-default.log` / `clippy-final.log` |
| Rustdoc，警告视为错误，默认/全特性 | 通过 | `rustdoc-default.log` / `rustdoc.log` |
| rustfmt / diff whitespace | 通过 | `fmt-final.log`；`git diff --check` |
| 默认依赖隔离 | 无 tokenizers/onig/tiktoken-rs/xz2 普通依赖 | `dependency-tree.log` |
| 离线打包及包内编译 | 667 文件，21.0 MiB，压缩 11.0 MiB；通过 | `package.log` |

初次全量中本地回环绑定被沙箱拒绝，放行本机 127.0.0.1 测试后最终全量通过。初期失败日志保留；不能将旧夹具失败或环境限制计为最终未修复代码错误。打包后仅补充本轮状态记录，无生产代码变化。

收尾结论：当前有限、契约明确的功能清单和本地交付检查已完成，没有遗留已确认编码任务。P0–P7 中宽泛领域描述仍是后续审计范围，Unknown 单元格不是同等数量的任务。继续单列以下证据边界：

1. 未建立契约：OpenAI Speech SSE 文本/音频对齐、Z.AI 国际 TTS、Anthropic 仅消息内按值定义的 Tool Search 可发现性、xAI 会话条目删除/截断的完整请求字段。不猜测协议。
2. 未执行真实账户/区域验收，在线状态仍为 `not_run`。此前自动审批拒绝使用环境 XAI_API_KEY 的外发请求，后续必须获得明确授权；本轮没有重试或绕过。
3. 宿主职责：跨作业文件清理策略、跨进程单次消费、凭证刷新、工具执行与设备层。OpenAI Vector Store 搜索的 `next_page` 缺少官方请求游标，不虚构续页操作。

所有验证进程和本轮工作子任务均已结束。无手动 compact 工具；更新交接记录不代表执行了压缩。
