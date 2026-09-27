# Qwen Chat prompt caching

[中文](qwen-prompt-cache.md)

Qwen Chat Completions accepts explicit message breakpoints through `ChatRequest.prompt_cache`. The [Model Studio Context Cache contract](https://help.aliyun.com/en/model-studio/context-cache), checked on 2026-09-25, places `cache_control: {"type":"ephemeral"}` on message content blocks. Its fixed five-minute lifetime refreshes on hits. Implicit caching is provider-managed, without an enable/disable switch or caller-selected TTL; implicit and explicit modes are exclusive.

```rust
use lingxi_llm_client::protocol::{CacheBreakpoint, CachePosition, CacheTtl};

# fn configure(request: &mut lingxi_llm_client::protocol::ChatRequest) {
request.prompt_cache.breakpoints = vec![
    CacheBreakpoint {
        position: CachePosition::System { index: 0 },
        ttl: CacheTtl::FiveMinutes,
    },
    CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    },
];
# }
```

Leave the default empty policy for implicit caching. The adapter rejects `automatic: Some(...)`, one-hour TTLs and independent Tool-definition markers. Up to four unique breakpoints may target nonempty system/text/tool-result content or an image. Thinking, tool-use, document and out-of-range positions fail validation.

Encoding preserves original system boundaries and separators, user/assistant content, and tool-result call IDs. Marked single strings become content arrays. Splitting a tool result into its own wire message does not move its breakpoint. The caller's request is unchanged.

## Region, deployment scope and preflight

Explicit policies require a documented Model Studio HTTPS Chat root and exact request model ID. Beijing defaults to `china_mainland`; Singapore defaults to `international`. Other regions require the actual deployment scope in profile `extra.qwen_cache_deployment_scope`:

| Region | Accepted deployment scopes |
| --- | --- |
| US Virginia | `global`, `us` |
| Hong Kong | `global`, `hong_kong` |
| Frankfurt | `global`, `eu` |
| Tokyo | `global`, `japan` |

For example, a US Global deployment uses `extra = { qwen_cache_deployment_scope = "global" }`. This is client validation metadata and is not sent to the provider. Model lists from different scopes are never combined. Unknown models, aliases, proxy roots and unverified combinations fail locally. Configuration does not establish account access; callers provide credentials for the correct region and deployment.

Validation precedes attachment resolution, uploads, authentication and transport. Top-level native overrides through `extra.body.cache_control`, `prompt_cache_options`, `prompt_cache_key` or `prompt_cache_retention` are rejected. This adapter does not enable Responses session caching or another vendor's similarly named controls.

The provider's 1,024-token minimum and 20-content-block lookback affect cache hits. The client neither estimates the threshold from character counts nor promises a cache write or hit.

## Usage and validation

Chat `prompt_tokens` includes cache reads and writes. `prompt_tokens_details.cached_tokens` becomes cache-read usage; `cache_creation_input_tokens` becomes cache-write usage. Subtract both to obtain ordinary input: 2,000 total prompt tokens, 1,200 reads and 500 writes leave 300 ordinary tokens. The one-hour write counter stays zero; absent pricing remains unknown.

Qwen streaming requests set `stream_options.include_usage=true`; contradictory settings fail preflight. Complete and streaming responses share normalization. A later explicit zero replaces earlier write usage. Out-of-range, malformed or conflicting counters do not produce a complete billable report.

`tests/qwen_prompt_cache.rs` uses offline fixtures, fragmented SSE and side-effect-rejecting mocks. It also checks encoded length against actual bytes. No live provider account has been tested.
