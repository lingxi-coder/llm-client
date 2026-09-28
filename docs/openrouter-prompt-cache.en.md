# OpenRouter Prompt Caching

OpenRouter provider prompt caching reuses model-side prompt prefixes. It is separate from gateway response caching configured with `RequestOptions.openrouter_response_cache`. Gateway response caching uses `X-OpenRouter-Cache`, TTL, and clear headers; a hit can return a complete response without calling the upstream provider. Provider prompt caching is handled by the model provider and is reported through cached-token usage fields. An empty `ChatRequest.prompt_cache` leaves each provider's default behavior in place.

This client encodes an explicit `ChatRequest.prompt_cache` policy only on the official OpenRouter Chat route, `https://openrouter.ai/api/v1/chat/completions`, and selects an adapter by the requested model. It does not send Anthropic `cache_control` fields to arbitrary OpenRouter models.

The supported combinations are:

- `anthropic/...` models support top-level automatic caching and explicit text-block breakpoints. TTL is five minutes or one hour, with at most four breakpoints including the automatic breakpoint. One-hour breakpoints must come before five-minute ones.
- The Alibaba models listed by OpenRouter support one explicit text-block breakpoint with a fixed five-minute TTL: `deepseek/deepseek-v3.2`, `qwen/qwen3-max`, `qwen/qwen-plus`, `qwen/qwen3.6-plus`, `qwen/qwen3-coder-plus`, and `qwen/qwen3-coder-flash`. `automatic` and other TTLs fail before dispatch.
- `openai/gpt-5.6` and later GPT models support explicit cache options with a 30-minute TTL. Automatic breakpoints encode as `prompt_cache_options: { mode: "implicit", ttl: "30m" }`. Explicit breakpoints use `mode: "explicit"` and OpenRouter's documented block-level `cache_control` marker, which OpenRouter translates to the OpenAI provider's `prompt_cache_breakpoint`. Implicit mode allows up to three additional explicit breakpoints; explicit-only mode allows four.

Breakpoints currently accept only nonempty text blocks. They cannot mark tool definitions, images, audio, or file blocks. OpenAI GPT-5.5 and earlier, Grok, automatically cached DeepSeek, automatically cached Gemini, and other unlisted models need no request controls; an explicit policy for them is rejected. OpenRouter does not document a caller-settable TTL or breakpoint for these automatic caches.

```rust
use lingxi_llm_client::protocol::{CacheBreakpoint, CachePosition, CacheTtl, PromptCachePolicy};

let policy = PromptCachePolicy {
    automatic: None,
    breakpoints: vec![CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::ThirtyMinutes,
    }],
    ..PromptCachePolicy::default()
};
```

`CacheTtl::ThirtyMinutes` is accepted only for OpenAI GPT-5.6 and newer through OpenRouter. Direct Anthropic, Qwen, MiniMax, and other routes that do not support this TTL reject it before a network request. Explicit OpenAI markers use Chat Completions content blocks; they are never sent to older models by automatic routing.

OpenRouter's [Prompt Caching guide](https://openrouter.ai/docs/guides/best-practices/prompt-caching) distinguishes upstream automatic caching from explicit breakpoints. Its [Chat Completions API reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion) defines top-level `cache_control`, block-level `cache_control`, and `prompt_cache_options` for OpenAI GPT-5.6+. OpenRouter's [Response Caching guide](https://openrouter.ai/docs/guides/features/response-caching) separately documents gateway cache request headers and `HIT`/`MISS` response status. The [OpenAI Prompt Caching guide](https://developers.openai.com/api/docs/guides/prompt-caching) confirms `30m` is the only GPT-5.6+ TTL and documents the four-cache-write limit.

These capabilities are verified with request-encoding tests; no live OpenRouter account was used.
