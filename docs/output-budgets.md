# 输出预算与 provider 语义

本文记录 2026-10-14 核对的官方 API 规则和当前工作树修复。工作树变更不代表移动端已安装的二进制已经更新。

## 默认策略

默认请求在额度可选的 API 上省略输出预算，采用 provider 的官方默认。`ChatRequest.max_tokens = None` 始终保留自动意图；每次实际连接准备时，SDK 单独克隆请求解析额度，切换连接后重新解析。显式预算保留为硬上限，只能向下受选中模型已知上限约束。

Messages 及 Claude 托管接口的额度必填，使用明确匹配模型的现有应用默认，或模型行 `metadata.applicationOutputTokens`（TOML 为 `application_output_tokens`）。兼容接口的内置非 Claude 模型将原有逐模型应用策略显式存入目录；这些数字不是 provider 默认，不能用于额度可选的连接。应用默认必须为正且不超过该行已知上限；未知模型缺少默认时返回配置错误。没有全局 `4096` 回退，也不把模型上限直接当默认。

目录 context 数值本身不证明输入与输出共享窗口，因此不再从所有模型额度中统一扣除输入估算和固定余量。上下文错误仍由 Harness 的上下文恢复处理；预算解析不伪造 1 token 额度。

请求编码后、钩子执行后以及 draft 封存时检查预算字段一致性，拒绝隐藏的预算覆盖。ChatGPT OAuth／Plan 原生 Responses 接口不接受硬输出预算：自动请求省略字段，显式预算返回 `UnsupportedCapability`，不再静默忽略上限。

主回合保留环境／调用者的显式额度。side-query API 使用 `Option<u32>`；compaction 和 fork prompt suggestion 未指定额度时保留自动意图，简短后台建议、摘要和分类等任务的显式小额度保留。

## 已核对的参数差异

| Provider／接口 | 输出预算参数 | 官方规则与实现选择 |
| --- | --- | --- |
| DeepSeek Chat | `max_tokens` | 默认 thinking 模式为 64K，`max` effort 为 128K；非 thinking 为 8K。不能用通用 32k 覆盖这些规则。API 参数上限为 393,216 token（384 × 1024）。`length` 是额度截断，与 `aborted`、`insufficient_system_resource` 不同。 |
| OpenAI Responses／Chat | `max_output_tokens`／`max_completion_tokens` | 预算包含 reasoning、可见输出及非可见格式 token。 |
| Anthropic Messages，含云托管变体 | `max_tokens` | 参数必填。已知模型使用经校验的应用默认；显式预算优先。manual thinking 预算须低于有效输出预算（符合 interleaved thinking 例外时除外）。 |
| Gemini GenerateContent／Interactions | `maxOutputTokens`／`max_output_tokens` | 包含 thinking token。部分新模型另有单次请求上限和 long-decode／continuation 规则，不能把续接规则与硬截止混为一谈。 |
| Kimi 当前 Chat 接口 | `max_completion_tokens` | 官方已弃用 `max_tokens`；当前 Kimi、Kimi International、Kimi Code Chat 配置使用新字段。输入加预算必须满足所选模型的上下文限制。 |
| Qwen Chat | 按模型选择 | 官方列出的 Max 3.7+、Plus／Flash 3.5+、DeepSeek V4 Pro／V4.1 Flash 使用总输出参数 `max_completion_tokens`。其他系列仍按已公布的 `max_tokens` 接口编码，不能把答案预算与 thinking 预算混用。 |
| MiniMax Chat | `max_completion_tokens` | 新接入使用此参数，thinking 也占用生成额度；已有 preset 已使用正确字段。 |
| Z.AI Chat | `max_tokens` | 官方 GLM 系列提供独立输出上限，按所选模型目录行限制。 |
| Grok 当前 Responses | `max_output_tokens` | 官方明确只限制可见答案，reasoning 与 function-call token 不计入此参数。不能套用 OpenAI 的总生成 token 语义。 |
| OpenRouter | `max_tokens`／`max_completion_tokens` | 上限受剩余上下文约束；reasoning 计数取决于上游 provider。缺省参数转交上游默认，不提供通用固定值。 |

通过同一协议接入的 GitHub Copilot、自定义 endpoint 和其他兼容路由仍使用其选中目录行；不把原厂的模型上限强加到托管路由。

## 官方来源

- [DeepSeek Chat API](https://api-docs.deepseek.com/api/create-chat-completion)、[模型与价格](https://api-docs.deepseek.com/quick_start/pricing)
- [OpenAI reasoning 预算](https://developers.openai.com/api/docs/guides/reasoning)
- [Anthropic stop reasons 与截断工具调用](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [Gemini thinking 与输出额度](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [Kimi Chat API](https://platform.kimi.ai/docs/api/chat)、[Kimi Code completion 字段升级](https://www.kimi.com/code/docs/en/kimi-code-cli/release-notes/changelog.html)
- [Qwen Chat API](https://docs.qwencloud.com/api-reference/chat/openai-chat)
- [MiniMax OpenAI SDK 接口](https://platform.minimax.io/docs/api-reference/text-openai-api)
- [Z.AI Chat API](https://docs.z.ai/api-reference/llm/chat-completion)
- [Grok 输出额度](https://docs.x.ai/developers/model-capabilities/text/best-practices)
- [OpenRouter 参数](https://openrouter.ai/docs/api_reference/parameters)

## 截断恢复

流式解析器记录不完整工具的块序号、名称、ID 和参数字节数，并继续读取后续完整工具、终止原因和 usage。只有完整合法的调用进入执行队列。`max_tokens`／`length`、正常完成但工具参数不合法、资源不足、主动中断和没有终止事件分别处理；格式错误不再遮蔽已知的终止原因。

在没有接受任何工具、没有原生托管工具效果、usage 完整且实际发出额度与选中连接上限可确认时，允许一次提高额度并重试原请求。下一额度最多为原额度／已消费输出的两倍，受选中模型输出上限、已确认的共享上下文余量和显式用户额度约束。没有额度事实、预算来自显式调用者限制、路由已变化或存在外部效果时不自动增加。

原请求快照保留模型、连接、系统提示、消息、工具及缓存策略；提高预算重试不把部分参数拼入新消息。完整工具 A 和不完整工具 B 同时出现时，保留 A 的执行结果，仅提示重新生成 B 的完整调用，避免重复执行 A。不能恢复原请求的文本截止采用语义继续，工具参数截止采用完整调用重生成。提高预算、格式修复及语义继续共同使用每次用户回合最多三次恢复动作，交替失败也不能重置额度。

这不等于所有 provider 支持原生字节续接。Anthropic 官方对截断工具参数建议提高预算重试原请求；DeepSeek beta prefix completion 是文本前缀接口，不能据此拼接工具参数。

### 原生 Gemini long-decode 的待完成部分

当前代码没有实现 Gemini 4 的 `CONTINUATION` 自动 hop、加密 `continuationToken` 的作用域绑定或断流 checkpoint 提交。当前有界语义恢复不应称为 Gemini 原生续接。

官方 long-decode 文档规定续接必须复用原 contents、systemInstruction、tools 和 generationConfig，发送顶层 continuationToken；`CONTINUATION` 保持总额度，而带 token 的 `MAX_TOKENS` 续接需要提高总额度。流式／非流式单 hop、usage 累计及断流 checkpoint 都需单独实现。该页的首次非流式额度限制与 REST 示例存在冲突，必须先用真实接口确认，不能按其中一个示例猜测。本机没有可用于核对的 Gemini API 凭证。

### 本次验证及交付边界

SDK 检查覆盖所有目录行的可选预算省略、必填预算解析、同名模型不同连接、Kimi／Qwen 字段、Native ChatGPT 不支持显式预算、钩子封存与坏工具参数诊断。Harness 检查覆盖真实文件写入、已完成结果保留、冻结原请求提高预算及统一恢复上限。

iOS 使用独立产品源码快照和本地 SDK／Harness 路径构建验证，不修改并行中的产品 UI。为匹配当前 bindings，验证快照采用了产品工作树中已有的 `IOSAudioServiceCallback.swift` 适配，不改变原文件。

2026-10-14 验证结果：

- SDK：families 37、inference_controls 19、output_budgets 11、prepared_calls 32，共 99 项通过。
- llm-runtime：请求构建 20 项、stream_accumulator 26 项通过。
- orchestrator：mobile_truncated_tool_test 5 项、streaming_max_tokens_recovery_test 2 项通过；包含真实 Write 工具、混合完整／坏调用、正常结束／额度截止／缺失终止、冻结原请求及三次统一恢复上限。两项旧流式回归需要 `RUST_MIN_STACK=33554432`，默认测试线程栈会溢出，不应将其报告为默认环境全绿。
- 最终源码的 device-only framework 和 Swift bindings 重新生成成功；`LingxiCodeFull / FullDebug / generic iOS` 完整编译签名成功。SwiftPM 官方二进制下载超时后，复用了已核对 owning package revision 的九个本机缓存二进制。
- 应用位于 `/tmp/mobile-budget-ios-derived/Build/Products/FullDebug-iphoneos/LingxiCode.app`；bundle 为 `com.lingxi.code.full`，版本 2.4.1 (8721)，可执行文件 SHA-256 为 `c9adf1c0109eadc81cb0b809da6acbd10ea60a5e10f12a0b54b336d31a6166ad`。
- 安装未成功：CoreDevice error 1011，刷新设备列表后 iPhone 状态为 unavailable。没有新版本安装或原始动画场景真机通过的证据。

本次验证不是正式 Git 依赖 pin 或发布：产品正常 Cargo 配置仍使用原有固定版本，本地修复只通过独立验证目录的 path patch 接入。所有 provider 的真实接口回放、Gemini 原生续接、正式依赖集成及真机验收尚未完成。截图对应旧失败没有保留原始 finish reason／wire 预算，因此不能反推该次必然由某个输出额度导致。

- [Anthropic 不完整工具调用](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [DeepSeek 文本 prefix completion](https://api-docs.deepseek.com/guides/chat_prefix_completion)
- [Gemini long-decode](https://ai.google.dev/gemini-api/docs/generate-content/long-decode)
