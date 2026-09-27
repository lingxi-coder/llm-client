# OpenRouter 提示缓存

OpenRouter 的提供方提示缓存复用模型端的 prompt 前缀；它与 `RequestOptions.openrouter_response_cache` 控制的网关响应缓存是两套机制。网关响应缓存使用 `X-OpenRouter-Cache`、TTL 和清除请求头，命中时可返回完整响应且不调用上游。提供方提示缓存则由模型提供商处理，并通过 token usage 中的缓存字段报告；空 `ChatRequest.prompt_cache` 保留各提供商自己的默认行为。

本客户端只把明确的 `ChatRequest.prompt_cache` 策略编码到官方 OpenRouter Chat 路由 `https://openrouter.ai/api/v1/chat/completions`，并按请求模型限制适配器。它不会把 Anthropic 的 `cache_control` 字段发送给任意 OpenRouter 模型。

目前支持的组合：

- `anthropic/...` 模型支持顶层自动缓存以及文本块显式断点。TTL 为五分钟或一小时；最多四个断点（包含自动断点）。一小时断点需排在五分钟断点之前。
- OpenRouter 文档列出的 Alibaba 缓存模型支持一个文本块显式断点，固定五分钟：`deepseek/deepseek-v3.2`、`qwen/qwen3-max`、`qwen/qwen-plus`、`qwen/qwen3.6-plus`、`qwen/qwen3-coder-plus` 和 `qwen/qwen3-coder-flash`。`automatic` 和其他 TTL 会在发送前拒绝。
- `openai/gpt-5.6` 及后续 GPT 模型支持 30 分钟 TTL 的显式缓存选项。只使用自动断点时，编码为 `prompt_cache_options: { mode: "implicit", ttl: "30m" }`；显式断点使用 `mode: "explicit"` 并在文本块处写入 OpenRouter 文档的 block-level `cache_control` 标记，由 OpenRouter 转换成 OpenAI 提供方的 `prompt_cache_breakpoint`。隐式模式最多再放三个显式断点；显式模式最多四个。

断点目前仅接受非空文本块，不能标记工具定义、图片、音频或文件块。OpenAI GPT-5.5 及更早模型、Grok、DeepSeek 自动缓存、Gemini 自动缓存，以及未列出的其他模型不需要这些请求控制；为它们设置显式缓存策略会被拒绝。OpenRouter 文档没有为这些自动缓存提供可由本策略设置的 TTL 或断点。

```rust
use lingxi_llm_client::protocol::{
    CacheBreakpoint, CachePosition, CacheTtl, PromptCachePolicy,
};

let policy = PromptCachePolicy {
    automatic: None,
    breakpoints: vec![CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::ThirtyMinutes,
    }],
    ..PromptCachePolicy::default()
};
```

`CacheTtl::ThirtyMinutes` 只适用于 OpenRouter 上 GPT-5.6 及更新的 OpenAI 模型；直接 Anthropic、Qwen、MiniMax 或其他不支持此 TTL 的路由会在网络请求前拒绝。显式 OpenAI markers 使用 Chat Completions 内容块，自动路由到旧模型不会发生。

OpenRouter 的 [提示缓存指南](https://openrouter.ai/docs/guides/best-practices/prompt-caching)说明哪些上游会自动缓存、哪些要求显式断点；[Chat Completions API 参考](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)定义顶层 `cache_control`、block-level `cache_control` 和 OpenAI GPT-5.6+ 的 `prompt_cache_options`。OpenRouter 的[响应缓存文档](https://openrouter.ai/docs/guides/features/response-caching)单独列出网关缓存请求头和 `HIT`/`MISS` 响应状态。[OpenAI 提示缓存指南](https://developers.openai.com/api/docs/guides/prompt-caching)确认 GPT-5.6+ 的唯一 TTL 是 `30m`，并说明最多四次缓存写入。

这些能力通过请求编码测试验证；没有调用真实 OpenRouter 账户。
