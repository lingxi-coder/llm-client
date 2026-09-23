# 项目审查与修复记录

[English](review.en.md)

初次审查覆盖客户端路由与故障转移、认证与传输边界、内置 codec、流解析、模型目录、用量与价格计算，以及独立仓库的 API 文档构建。该轮修复保留现有接口形状，没有引入依赖。后续内置 HTTP 功能单独说明如下。

## 已修复的问题

| 优先级 | 问题与影响 | 修复位置 |
| --- | --- | --- |
| 高 | `HttpRequest` 的 Debug 输出包含认证头、URL 凭证和提示词，日志可能泄露这些数据 | `src/transport.rs`：隐藏 URL、头值、body，只保留请求方法、头名、长度和超时 |
| 高 | `stream()` 使用默认 options 时编码成非流式请求；`complete()` 也可能误发流式请求 | `src/client/failover.rs`：由高层方法决定模式，不修改调用方 options |
| 高 | Chat/Gemini 的流内错误被忽略，Responses failed 事件丢失错误分类 | `src/codecs/*/stream.rs`：识别错误并保留语义分类 |
| 高 | 缺少终止标记的 EOF 被当成正常完成，可能让截断输出进入后续处理 | 四个基础 stream decoder：返回 `StreamInterrupted`，保留正常终止路径 |
| 中 | `profile/model` 限定引用可能选中同名 group 的其他连接 | `src/client/resolve.rs`：精确 profile 名优先 |
| 中 | Anthropic 模型目录缺少版本头 | `src/directory/anthropic.rs`：发送 `anthropic-version`，支持配置版本 |
| 中 | 部分 codec 把非 2xx、非法 JSON 或缺少必要字段的响应当成空成功 | Anthropic/Gemini/Responses decode：验证状态和响应形状 |
| 中 | 文本文档未经 Base64 编码；Responses URL 文件使用了错误字段 | Gemini/Responses encode：编码文本，URL 使用 `file_url` |
| 中 | SSE 的 CR 行尾、跨 chunk CRLF 和无冒号 data 字段解析错误 | `src/framing/sse.rs`：规范化行尾，保留空 data 行 |
| 中 | Gemini token 加法可能溢出；缓存读写合计超过输入仍被认为有效 | Gemini decode、`src/client/usage.rs`：防溢出并验证计数一致性 |
| 中 | 非法费率、倍率或金额溢出仍返回成功估算 | `src/client/pricing.rs`：返回 `CostUnavailable` |
| 中 | 峰值时间窗的大数字可能 panic；无法表示结束于午夜的窗口 | `crates/agent-api/src/protocol/provider.rs`：先验证再计算，支持结束时间 `24:00` |
| 中 | 失效的 Rustdoc 内部链接使严格文档构建失败 | 共享 protocol 文档：修正不存在的链接 |

## 文档与验证

- [API 文档](api.md) 覆盖接入、请求、事件、配置、认证、传输、目录、费用、错误与扩展点。
- 指南通过 `include_str!` 纳入 Rustdoc，Rust 示例由 doctest 编译检查。
- 行为修复均配有回归测试；测试使用模拟传输，无需真实 API key。
- 初次审查验证（不包含后续内置 HTTP 功能）：228 项单元/集成测试及 4 个文档示例通过；相较基线新增 24 项行为回归测试。格式检查、Clippy（警告视为错误）和严格 Rustdoc 构建通过，文档本地链接目标均存在。

```sh
cargo test --workspace --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
```

## 后续修复与仍需由宿主处理的边界

- 本轮修复将首连接凭证限定在首连接；备用连接需由宿主按 profile 提供独立凭证。完整和流式响应会暴露实际成功连接，供调用方按该连接估价。
- 本次后续修复已处理 Gemini 工具 ID 和签名、OpenAI 消息回放与截断响应、请求时限、SSE/AWS 帧大小边界及峰值价格配置验证。这些修复不计入前述初次审查的测试数量。
- 后续已实现内置 HTTP 客户端；WebSocket 调度、OAuth 刷新和目录自动合并仍未提供。
- 没有进行真实 provider 网络请求、真实账单校验或所有 provider 配置的在线可用性验证；静态目录仍需要维护。

## 后续：内置 HTTP 客户端

`LlmClientBuilder::new(&profiles)?` 默认提供内置 `HttpTransport`、`SystemClock` 和 API key / Bearer 认证器。移除 `LlmServices`，调用方无需组装服务；可用 `with_transport()` 注入自定义传输，或用 `with_clock()` 设置测试时钟。这是构建器接入方式的变更，原先的服务注入调用应迁移至上述入口。内部使用 reqwest 0.12 与 Rustls，调用方只需提供 Tokio 运行时和请求凭证，无需自行导入 reqwest 或编写 HTTP 适配器。

默认禁止重定向和自动重试，连接超时为 30 秒；请求指定的总超时覆盖流读取。HTTP 状态和错误 body 保留给 codec 处理，网络错误消息避免暴露敏感请求数据。详见 [传输接口](api.md#传输接口)。上面的测试数量记录初次审查结果，新增功能需以当前测试运行结果为准。

## 协议核对来源

- [Claude Models API](https://platform.claude.com/docs/en/api/models/list)：模型列表请求头。
- [OpenAI File inputs](https://developers.openai.com/api/docs/guides/file-inputs)：文件 URL 与 Base64 字段。
- [Gemini generateContent](https://ai.google.dev/api/generate-content)：inlineData 数据格式。
- [WHATWG Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)：行尾和字段解析。


本次内置 HTTP 与时钟变更验证：236 项单元/集成测试及 4 个文档示例全部通过；格式检查、Clippy（所有 targets，警告视为错误）和严格 Rustdoc 通过。新增 7 项本地 TCP HTTP 测试覆盖请求/响应、重定向、即时流式读取、总超时、断流与内置认证，并用固定时钟验证不同时段的费用。未使用真实 provider 或付费请求；HTTPS 信任链未做线上验证。

HTTP 实现依照 [reqwest 0.12 ClientBuilder 文档](https://docs.rs/reqwest/0.12.28/reqwest/struct.ClientBuilder.html) 设置连接超时、关闭重定向与自动重试。TLS 连接错误目前归入 `Transport`，不单独细分证书错误。

## 设计审查后的接口与状态修复

本轮保留 codec、认证器、传输和目录解析器的既有分层，修复请求目标与凭证绑定、协议终态和配置同步边界；没有增加依赖。

- 增加 `complete_in`、`stream_in`、`web_search_in` 和 `web_search_stream_in`。这些接口显式选择起始 profile 或组，并用同一条已解析路由执行请求。未限定的原生模型 ID 与 `profile/model` 指向不同目标时返回 `AmbiguousNativeAndQualified`，避免 OpenAI 与 OpenRouter 同时配置时误把主凭证发送到另一连接。合法且没有歧义的含 `/` 原生 ID 保持可用。
- Anthropic 流式解码保留 `redacted_thinking`；用量完整性需要最终输出计数和数值自洽，初始 seed 或单独的结束标记不能把部分用量提升为最终用量。
- Gemini 提示词拦截和 Responses 拒绝内容在完整响应、流式响应中保持一致的拒绝语义；截断与 provider 错误仍保留原有优先级。
- 流式入口收到非成功 HTTP 状态后，即使错误体中断，也保留 HTTP 状态、已收集内容和 `Retry-After`，按原策略判断故障转移。已经返回成功流后的中断仍不会自动重放。
- Anthropic 目录缺少布尔类型的 `has_more`，或声明 `has_more: true` 却缺少有效 `last_id` 时返回错误，避免静默接受不完整目录。
- 客户端的持久化覆盖、缓存和跟踪状态收拢到内部 `ProviderStore`，与用于请求的有效 profile 快照分开。`prepare_provider_sync` 创建不借用客户端的操作，`fetch` 获取目录，`apply_provider_sync` 校验并提交；同步涉及的文件锁和文件 I/O 在 Tokio blocking 线程池执行。提交前核对新鲜的磁盘状态、连接配置和配置目录代次。
- `lingxi-agent-api` 的默认 `agent` feature 保留原完整 API；LLM 客户端关闭该 feature，并通过 `lingxi_llm_client::protocol` 重导出协议。CI 单独验证最小 feature 集，避免工作区 feature 合并掩盖问题。
- 模型新增可选的三态能力信息，区分未知、支持和不支持。保留旧布尔字段：明确的新元数据优先，旧 `true` 可作为支持依据，旧 `false` 保持未知。新增目录模型未取得能力事实时保持未知；能力元数据仍供宿主判断，不引入新的运行时拒绝。

迁移时，原有 JSON 配置可继续读取；Rust `ModelProfile` 字面量需补 `capability_support: None`。依赖未限定模型名的调用方如遇歧义，应改用显式 profile 的 `_in` 接口。直接使用 `lingxi-agent-api` 的旧导入仍然有效。

配置同步的获取阶段可以取消而不写盘。提交进入文件写入阶段后，取消仍可能留下已更新的文件和未更新的内存快照；宿主在该情况下应重新加载配置。同步配置管理方法仍保持阻塞接口，异步宿主应选择合适的执行环境。

本轮验证：335 项单元/集成测试与 12 项文档测试通过；格式检查、Clippy（警告视为错误）、严格 Rustdoc 通过。共享协议的默认和最小 feature 构建/测试通过，最小 feature 的严格 Rustdoc 也通过。共享 crate 的离线打包及包内构建验证通过；客户端打包列表已排除本地 `.omx` 运行状态。本轮仅使用模拟传输和回环 HTTP，未请求真实 provider 或验证线上费率。


## 第二轮整体验证后的修复

本轮继续使用 GPT-6 Luna（max）实施，并由独立审查者复核。六项问题均先通过回归用例或独立下游工程复现；没有增加运行时依赖。

- 非流式和流式入口共用有界错误体读取逻辑。已收到非 2xx HTTP 状态后，连接中断或错误体超时不再抹掉状态、响应头与 `Retry-After`；最多保留 64 KiB。成功响应仍要求 body 完整，总请求时限继续限制备用连接尝试。
- Responses 普通响应按序保留所有 reasoning summary 文本块，与流式摘要内容一致。
- Anthropic 用量校验覆盖独立缓存读写计数：字段存在时必须是无符号整数，归一化总量不能溢出；允许缺省、数值零和高于未缓存输入的有效缓存计数。
- Gemini 使用明确的 `supportedGenerationMethods` 排除无法调用 `generateContent` 的模型；静态目录也记录相应方法元数据。未提供方法信息仍表示未知，不能清除已有不支持的事实。同步按连接保存排除信息，防止旧模型从缓存或后续白名单扩展中恢复；明确支持生成的新观察或显式配置替换可以解除排除。目录未列出的其他模型仍按原有合并语义保留。
- Gemini 分页 token 的错误类型使同步失败，旧配置保持不变；缺省、空字符串和兼容的 null 仍表示末页。目录 trait 增加带默认实现的可选方法以传递操作兼容性事实，保留原 `decode_page`、`ModelPage` 和 `LiveModel` 的既有用法。
- 文档统一通过 `lingxi_llm_client::protocol` 导入类型。新增独立的 `tests/downstream-docs` 工程，用 README 声明的直接依赖编译 README 和两份指南的 Rust 示例；CI 单独运行该检查，避免工作区开发依赖掩盖下游编译错误。

最终验证：352 项单元/集成测试和 12 项工作区文档测试全部通过；独立下游工程的 13 个文档示例通过，相比本轮修复前新增 17 项行为回归测试。格式检查、Clippy（所有 targets、警告视为错误）、严格 Rustdoc、共享协议最小 feature 测试和客户端独立构建通过。共享 crate 离线打包及包内构建验证通过；客户端包列表不包含本地运行状态、下游测试工程或构建产物。独立复核确认六项修复没有剩余实质问题。

同一 profile 的目录观察没有版本号，并发获取结果按宿主提交顺序应用；需要最近发起的刷新优先时，宿主应串行刷新或在提交前丢弃过期结果。验证仅使用模拟传输与本地回环 HTTP，没有调用真实 provider 或核验线上账单。
