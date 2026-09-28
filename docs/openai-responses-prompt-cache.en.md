# OpenAI Responses prompt caching

Native OpenAI Responses cache fields live in `ChatRequest.prompt_cache` and are encoded only for the official `https://api.openai.com/v1` Responses route. `prompt_cache_key` can help cache routing or separate accounting; it does not guarantee a cache hit.

```rust
# use lingxi_llm_client::protocol::ChatRequest;
# let mut request: ChatRequest = serde_json::from_value(serde_json::json!({
#     "model":"gpt-5.6-sol",
#     "system":[{"text":"Stable system policy"}],
#     "messages":[{"role":"user","content":[{"type":"text","text":"Question"}]}]
# })).unwrap();
use lingxi_llm_client::protocol::{CacheBreakpoint, CachePosition, CacheTtl, OpenAiPromptCacheMode,
    OpenAiPromptCacheOptions, OpenAiPromptCacheTtl};

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

`prompt_cache_options` is supported only on GPT-5.6 and later. Its TTL sets the **minimum cache lifetime**; `30m` is currently the only supported value. Omitting `mode` keeps the API's implicit default. Implicit mode uses one breakpoint slot, leaving room for at most three explicit breakpoints. Explicit mode uses only caller breakpoints and allows up to four. Explicit mode with no breakpoints is valid and disables cache writes for that request.

`CachePosition::System` and `CachePosition::Message` map to `input_text` content. Top-level `instructions` cannot carry an explicit marker. When a system block is marked, the encoder converts all system text into a developer message at the start of input and marks the selected `input_text` part with `prompt_cache_breakpoint: { mode: "explicit" }`. Other requests keep using top-level `instructions`. Text tool results and replayable Responses input messages also map by their original request indices to supported `input_text` parts.

Images, files, audio, assistant output, function calls, and tool definitions cannot carry an explicit breakpoint. A tool result can be marked when its final content is nonempty `input_text`. OpenAI Responses breakpoints currently require the existing `CacheBreakpoint.ttl` to be `CacheTtl::ThirtyMinutes`; this field does not replace `prompt_cache_options.ttl`.

Do not use generic `PromptCachePolicy.automatic` in place of the native options. Its `CacheTtl` describes cache policy for other protocols and does not mean the minimum lifetime in Responses.

`prompt_cache_retention` sets maximum retention and is independent of the minimum lifetime set by `prompt_cache_options.ttl`. OpenAI's reference still documents this field and marks it deprecated; the client sends it only when the caller explicitly sets it, with no fallback between fields. GPT-5.5, GPT-5.5 Pro, GPT-5.6, and later models support only `TwentyFourHours`; GPT-5.4, GPT-5.2, GPT-5.1 Codex Max, GPT-5.1, GPT-5.1 Codex, GPT-5.1 Codex Mini, GPT-5.1 Chat Latest, GPT-5, GPT-5 Codex, and GPT-4.1 support `InMemory` and `TwentyFourHours`. Therefore GPT-5.6+ requests may explicitly send `prompt_cache_options.ttl` and `prompt_cache_retention` together, including with explicit breakpoints. Other models reject unsupported values before dispatch.

Raw profile `extra.body` cannot override `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, or `prompt_cache_breakpoint`; set them through the typed fields above. Other protocols and OpenAI-compatible services do not receive these native OpenAI fields.

This follows the [OpenAI Prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching) and [Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create).
