# 保留 Anthropic 原生内容

Anthropic Messages 响应中，`text`、`thinking`、`tool_use` 等已知内容仍解码为对应的通用类型。其他完整内容块会保存在 `ContentBlock::ProviderContent` 中，并标记为 `AnthropicMessages`。带原生 citations 的文本块也会按原样保留。这样可以在下一轮 Anthropic Messages 请求中回放 `server_tool_use`、`web_search_tool_result`、`web_fetch_tool_result`、代码执行结果、工具搜索结果以及未来新增的内容块。

`ProviderContent` 表示需要保留的 Anthropic 原生数据，不表示客户端工具调用。编解码器不会执行 Anthropic 托管的 web search、web fetch、代码执行或工具搜索，也不会把它们转换成 `ToolUse`。重放到其他协议会被拒绝；如果应用需要在本地执行某项操作，应使用单独声明的客户端工具及其宿主权限流程。

流式响应对未知顶层事件，以及原生或未知内容块的 start、delta、stop 帧，发出 `StreamEvent::ProviderEvent { protocol, payload }`。每一帧各自保留，重复帧也不会去重。该事件只用于观察和记录，不能转换成消息内容或交给客户端工具执行。若原生块完整结束且所有 delta 都可解释，编解码器会在 `content_block_stop` 后另外发出 `ProviderContent`。服务器工具的 `input_json_delta` 会拼成最终 `input`，并保留起始块里的其他字段。遇到未知 delta 或无效的分片 JSON 时，编解码器保留事件帧并抑制该块的 `ProviderContent`，以免回放不完整数据。等待关闭的原生块和服务器工具输入受 8 MiB 聚合缓冲上限约束。

流式 text block 的非空起始文本和后续文本片段都会通过 `TextDelta` 输出。`citations_delta` 到达时，编解码器会把该 block 的文本与按序收到的 citations 组合起来，并在 `content_block_stop` 时发出完整的原生 `ProviderContent`，同时保留起始块的其他字段。Anthropic 定义每个 `citations_delta` 携带一个加入当前 block 的 citation；字符、页码和内容块位置等 citation 对象会按原样保留。纯文本块不会额外生成原生内容块。

可观察的原生帧仍以 `ProviderEvent` 单独保留。对于 citations 在流中途才出现的 block，编解码器会在发现 citation 时按该 block 的接收顺序输出已缓存的 start/text/citation 帧；因此这些观察事件的输出时机可能晚于其他 stream 事件，不能据此推断整个响应的全局事件时序。遇到未知或格式错误的 delta 时，会保留帧并抑制该 block 的 replayable `ProviderContent`。web-search 归因仍通过 `WebSearch` 输出。该 codec 不承诺把所有 Anthropic 工具或 citation 格式转换成统一的执行接口。

直接消费 `ModelStream` 时可以处理每个流事件。结构化流收集器会把事件保留在 `StructuredStreamResult.events`；普通 `ChatResponse` 不包含原始事件转录，因此仅保存最终响应的调用方不会保留 `ProviderEvent`。

## 官方文档

- [Messages API](https://platform.claude.com/docs/en/api/http/messages)
- [Streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Citations, including citation streaming](https://platform.claude.com/docs/en/build-with-claude/citations)
