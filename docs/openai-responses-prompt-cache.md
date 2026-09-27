# OpenAI Responses 提示缓存

OpenAI Responses 的原生缓存字段位于 `ChatRequest.prompt_cache`，只会在官方 `https://api.openai.com/v1` Responses 路由编码。`prompt_cache_key` 可用于缓存路由或分开统计；它不保证缓存命中。

```rust
# use lingxi_llm_client::protocol::ChatRequest;
# let mut request: ChatRequest = serde_json::from_value(serde_json::json!({
#     "model":"gpt-5.6-sol",
#     "system":[{"text":"Stable system policy"}],
#     "messages":[{"role":"user","content":[{"type":"text","text":"Question"}]}]
# })).unwrap();
use lingxi_llm_client::protocol::{
    CacheBreakpoint, CachePosition, CacheTtl, OpenAiPromptCacheMode,
    OpenAiPromptCacheOptions, OpenAiPromptCacheTtl,
};

request.prompt_cache.prompt_cache_key = Some("support:customer-17".into());
request.prompt_cache.prompt_cache_options = Some(OpenAiPromptCacheOptions {
    mode: Some(OpenAiPromptCacheMode::Explicit),
    ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
});
request.prompt_cache.breakpoints = vec![CacheBreakpoint {
    position: CachePosition::System { index: 0 },
    ttl: CacheTtl::ThirtyMinutes,
}];
```

`prompt_cache_options` 仅支持 GPT-5.6 及更新模型。TTL 是缓存条目的**最短生命周期**；目前唯一支持值是 `30m`。省略 mode 会保留 API 默认的隐式模式。隐式模式的一个断点槽由 OpenAI 使用，最多再接受三个显式断点；显式模式只使用调用方断点，最多四个。显式模式可以没有断点，这会关闭本次请求的缓存写入。

`CachePosition::System` 和 `CachePosition::Message` 会映射到输入内容中的 `input_text`。顶层 `instructions` 不能带显式断点；请求标记 system 块时，编码器才会把所有 system 文本转成位于输入开头的 developer 消息，并在选中的 `input_text` 块上写入 `prompt_cache_breakpoint: { mode: "explicit" }`。其他请求继续使用顶层 `instructions`。文本工具结果和可重放的 Responses 输入消息也会按原请求索引映射到支持的 `input_text` 块。

图片、文件、音频、assistant 输出、函数调用和工具定义不能作为显式断点。工具输出只有在它的末尾内容是非空 `input_text` 时才可标记。OpenAI Responses 断点目前要求现有 `CacheBreakpoint.ttl` 为 `CacheTtl::ThirtyMinutes`；该字段不会替代新的 `prompt_cache_options.ttl`。

不要用通用 `PromptCachePolicy.automatic` 代替原生选项。它的 `CacheTtl` 表示其他协议的缓存策略，不等于 Responses 的 TTL 最短生命周期。

`prompt_cache_retention` 设置最大保留期，与 `prompt_cache_options.ttl` 设置的最短生命周期彼此独立、互不影响。OpenAI 文档仍标注了该字段及其弃用状态；客户端仅在调用方显式设置时发送，不会从一个字段回退到另一个字段。GPT-5.5、GPT-5.5 Pro、GPT-5.6 及更新模型仅支持 `TwentyFourHours`；GPT-5.4、GPT-5.2、GPT-5.1 Codex Max、GPT-5.1、GPT-5.1 Codex、GPT-5.1 Codex Mini、GPT-5.1 Chat Latest、GPT-5、GPT-5 Codex 和 GPT-4.1 支持 `InMemory` 与 `TwentyFourHours`。因此，GPT-5.6 及更新模型可在同一请求显式设置 `prompt_cache_options.ttl` 与 `prompt_cache_retention`，也可与显式断点并用。未列出的模型会在发送前拒绝该字段。

原始 profile `extra.body` 不能覆盖 `prompt_cache_key`、`prompt_cache_options`、`prompt_cache_retention` 或 `prompt_cache_breakpoint`；请通过上述类型设置。其他协议和 OpenAI 兼容服务不会接收这些 OpenAI 原生字段。

规则依据 [OpenAI Prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching) 与 [Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create)。
