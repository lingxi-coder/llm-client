# 项目审查与修复记录

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
