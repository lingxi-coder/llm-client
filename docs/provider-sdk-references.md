# Provider Client 的官方 SDK 参考

本项目用 Rust 实现 provider client，参考官方 SDK 的资源划分、协议类型、错误信息和流式生命周期。参考范围限于本库支持的能力，不等于复刻 SDK 全部端点，也不引入 Python／Node 运行时。统一入口负责路由和协议归一化，provider client 负责各自的请求、响应与事件语义，共享 transport 负责网络资源。

## 固定参考版本

核查日期：2026-09-27（America/Los_Angeles）。以下版本来自当次官方仓库 `releases/latest` 返回的发布页，完整 SHA 来自发布页链接的 commit；它们是本轮参考基线，不是本项目的运行依赖。后续核查应记录新的版本和 SHA，不能让 `main` 的变化静默改变验收依据。

| Provider | 官方 SDK 发布 | 完整 commit SHA |
| --- | --- | --- |
| OpenAI | [openai-python v3.19.2](https://github.com/openai/openai-python/releases/tag/v3.19.2) | [`d9f7b7a085d4f2173756590abeffeb7a8c7641dd`](https://github.com/openai/openai-python/commit/d9f7b7a085d4f2173756590abeffeb7a8c7641dd) |
| Anthropic | [anthropic-sdk-python v1.8.0](https://github.com/anthropics/anthropic-sdk-python/releases/tag/v1.8.0) | [`4421d56a4dd23550c7097c9b7ab5668bd11e09c4`](https://github.com/anthropics/anthropic-sdk-python/commit/4421d56a4dd23550c7097c9b7ab5668bd11e09c4) |
| Google GenAI | [python-genai v2.25.0](https://github.com/googleapis/python-genai/releases/tag/v2.25.0) | [`2faba3c07bcabaa662d8bc4804d2f3f1994648d9`](https://github.com/googleapis/python-genai/commit/2faba3c07bcabaa662d8bc4804d2f3f1994648d9) |

当次官方支持列表中，OpenAI 将 Rust `async-openai` 列为社区库；Anthropic 和 Google GenAI 的官方 SDK 列表均未列出 Rust。因此这里选择官方 Python SDK 作为可核查的实现参考，不将社区 Rust 库称作官方 SDK。来源：[OpenAI SDK 列表](https://developers.openai.com/api/docs/libraries)、[Anthropic SDK 文档](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)、[Google SDK 列表](https://ai.google.dev/gemini-api/docs/libraries)。这些在线文档会更新，固定实现依据以上表的版本为准。

## 资源、错误与流式边界

### OpenAI

根 client 将 `responses`、`chat.completions`、`files` 等组织为资源入口，资源对象使用同一根 client。Responses 与 Chat Completions 是同一 provider 下的不同协议资源，不应仅因请求格式不同就复制整套连接和认证设施。来源：[固定版本的 client](https://github.com/openai/openai-python/blob/v3.19.2/src/openai/_client.py)。

SDK 区分连接错误、HTTP 状态错误和超时，并在响应或状态错误上保留 request ID。流消费期间的错误向调用方传播；已经开始交付内容的流不会自动重放。项目应保留 provider 错误信息与请求标识，并在归一化时区分正常完成、协议错误和连接中断。来源：[固定版本的错误、请求标识和流说明](https://github.com/openai/openai-python/blob/v3.19.2/README.md#handling-errors)。

### Anthropic

`messages`、`models`、`files` 等资源引用同一 client；HTTP client、超时和默认请求配置由根 client 传入基础层。来源：[固定版本的 client](https://github.com/anthropics/anthropic-sdk-python/blob/v1.8.0/src/anthropic/_client.py)。

消息流 helper 在原始事件流之上累计消息，并负责关闭响应。原始事件解码、消息累计和统一事件映射应分层，避免所有调用都强制缓存完整结果。HTTP 状态错误另保留状态码、provider 错误类型和 request ID，不能只转换成一段错误字符串。来源：[固定版本的流 helper](https://github.com/anthropics/anthropic-sdk-python/blob/v1.8.0/src/anthropic/lib/streaming/_messages.py)、[错误类型](https://github.com/anthropics/anthropic-sdk-python/blob/v1.8.0/src/anthropic/_exceptions.py)。

### Google GenAI

`models`、`files`、`caches` 等资源共享 `BaseApiClient`，同步与异步入口也从同一配置构建；资源对象不各自创建身份配置和连接池。来源：[固定版本的 client](https://github.com/googleapis/python-genai/blob/v2.25.0/google/genai/client.py)。

生成内容与流式生成有独立入口，API 错误提供 code 和 message；Gemini Developer API 与云平台入口具有不同的身份和 endpoint 配置。项目应保留 Google 原生内容与事件语义，再转换到统一协议，不能仅套用 OpenAI 消息字段。来源：[官方 SDK 使用说明](https://github.com/googleapis/python-genai#generate-content-synchronous-streaming)、[错误说明](https://github.com/googleapis/python-genai#error-handling)。

## 本项目不照搬的默认行为

以下是 provider 重构的设计约束，官方 SDK 的便利默认值不会自动成为本库的新契约。

- **重试只由一层执行。** OpenAI 参考版本默认重试两次；Anthropic 参考版本的 `DEFAULT_MAX_RETRIES` 也是 2。Google 参考版本在未配置 `retry_options` 时仅尝试一次；提供选项但未指定 `attempts` 时才使用包含首次调用的 5 次尝试。不能把三者拼成叠加重试，也不能只读取 Google 的常量就宣称其默认重试五次。Provider client 保持单次发送，已有显式重试／failover 策略统一处理截止时间、可重放性和取消；流已经交付内容后不自动重放。来源：[OpenAI retries](https://github.com/openai/openai-python/blob/v3.19.2/README.md#retries)、[Anthropic constants](https://github.com/anthropics/anthropic-sdk-python/blob/v1.8.0/src/anthropic/_constants.py)、[Google retry_args](https://github.com/googleapis/python-genai/blob/v2.25.0/google/genai/_api_client.py)。
- **凭证和区域由调用上下文显式提供。** 官方 SDK 会读取环境变量，Google 还定义 `GOOGLE_API_KEY` 与 `GEMINI_API_KEY` 的优先级。本库的 provider client 不新增环境凭证发现，不在请求失败时尝试其他环境账号，不从语言或 IP 推断区域；共享 client 不意味着共享账号。来源：[OpenAI 初始化](https://github.com/openai/openai-python/blob/v3.19.2/README.md#usage)、[Anthropic 初始化](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#usage)、[Google 环境配置](https://github.com/googleapis/python-genai)。
- **只共享机制，保留 provider 语义。** 连接池、deadline、取消和字节流设施可以共用；路径、鉴权头、JSON、原生错误与事件归各 provider。相同协议的实现可以复用，但 provider 名称相近或宣称兼容不代表全部字段与能力等价。
- **不增加旧版兼容层。** 旧模块路径、旧 client 别名和旧配置迁移不会为了模仿官方 SDK 被重新加入；也不移植 Python 的同步／异步双接口、自动工具执行循环或完整生成器代码。工具执行与 agent 编排仍属于调用方。

## 验证证据的含义

| 证据 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| 官方文档／固定 SDK 源码 | 参考版本的协议形态和实现行为 | 本库已实现该能力、当前账号有权限 |
| 本地 fixture／mock transport | 请求路径、鉴权、序列化、错误归一化、事件解码与完成条件符合测试契约 | 官方服务接受请求、模型行为、真实网络与计费 |
| 授权后的 live 验收 | 指定时间、账号、区域、模型及操作真实成功 | 其他账号、区域、操作或硬件端到端均可用 |

每个 provider 的本地测试应覆盖普通响应、HTTP 错误、分块边界、流中错误、取消和不完整结束；具有工具调用或 usage 的事件需验证字段不会在归一化时丢失。参考版本升级应先检查协议差异，再更新 fixture 和期望值，不能仅改快照使测试通过。

本轮参考研究只读取公开文档和源码，没有携带凭证访问模型服务。新增引用或通过 mock 不会修改 `live_validation` 为通过；真实验收范围和结果另见[提供方真实验收](provider-acceptance.md)。
