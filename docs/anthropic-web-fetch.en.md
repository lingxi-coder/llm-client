# Anthropic Messages Web Fetch

`HostedTool::AnthropicWebFetch` declares a provider-executed fetch tool on the first-party Anthropic Messages or an explicitly identified Foundry Messages profile. Foundry requires hosting and the underlying Claude `model_id` on the matching `ModelProfile` row; the request still carries the deployment name. The client never downloads URLs locally, emits host tool calls, or automatically retries an uncertain execution. Preserve returned native blocks for continuation.

```rust
use lingxi_llm_client::protocol::{AnthropicWebFetchConfig, AnthropicWebFetchVersion, ChatRequest, HostedTool};
# fn configure(request: &mut ChatRequest) {
request.hosted_tools.push(HostedTool::AnthropicWebFetch(AnthropicWebFetchConfig {
    version: AnthropicWebFetchVersion::V20260318,
    max_uses: Some(3),
    allowed_domains: vec!["example.com".into()],
    citations: Some(true),
    max_content_tokens: Some(4096),
    ..Default::default()
}));
# }
```

The four versions are explicit capability choices; there is no automatic downgrade:

| Version | Controls |
| --- | --- |
| `V20250910` | Basic fetch |
| `V20260209` | Adds provider-side dynamic filtering |
| `V20260309` | Also accepts `use_cache` |
| `V20260318` (default) | Also accepts `response_inclusion` (`Full` or `Excluded`) |

Dynamic filtering implicitly enables provider-side `Code Execution`. Azure-hosted Foundry accepts only `V20250910` basic fetch. Anthropic-hosted Foundry accepts all versions, but dynamic filtering requires an underlying Claude 4.6+ or Mythos Preview model documented for that feature. The codec checks the explicit Foundry identity and does not require a separate `Code Execution` declaration. Azure-hosted Foundry does not expose `Code Execution` / `PTC`, so basic fetch also accepts direct callers only.

For newer versions that support dynamic filtering, setting `allowed_callers` to `Direct` disables it, so the dynamic-filtering model restriction does not apply. Azure-hosted Foundry still accepts only `V20250910`; final model acceptance for Anthropic-hosted deployments remains provider-authoritative.

All versions expose `allowed_callers`, `strict`, `defer_loading`, a five-minute or one-hour `cache_control`, and `url_sources`. The fetch-specific caller enum includes `Direct` and the documented Code Execution caller versions; it does not broaden the client-function PTC caller contract. Deferred fetch needs a supported way to surface the tool and cannot carry a cache marker. Cache markers share the request's four-marker and TTL-order checks. Strict fetch counts toward the twenty-strict-tool limit.

To add Web Fetch by value in a mid-conversation system message, set `inline_definition: true` on this config and include a `tool_addition` whose definition exactly matches `tool_value()` for the config. This placement is currently limited to the first-party Claude API. The typed config still controls version, model, hosting, source, and retry validation, and only the top-level Web Fetch entry is suppressed. If `cache_control` is configured, put its native marker in the inline definition; it is counted at that message position rather than as a synthetic top-level tool marker. The provider remains authoritative for whether this server tool type accepts by-value placement on a given API release.

URL filters preserve omitted defaults and explicit empty `Only`/`Except` lists:

```rust
use lingxi_llm_client::protocol::{AnthropicFetchToolReference, AnthropicFetchToolResultsSources, AnthropicFetchUrlSources, AnthropicFetchUserInputSources};
let sources = AnthropicFetchUrlSources {
    user_input: Some(AnthropicFetchUserInputSources::All),
    client_tool_results: Some(AnthropicFetchToolResultsSources::Only {
        tools: vec![AnthropicFetchToolReference::new("lookup")],
    }),
    server_tool_results: Some(AnthropicFetchToolResultsSources::None),
};
// Assign to config.url_sources; also declare the caller tool "lookup" in request.tools.
```

Named source references must be declared in the request's top-level tools. Implicit Code Execution and MCP names are not invented. Server-side URL eligibility remains Anthropic's decision; declaring a source filter does not cause a local fetch.

Allowed and blocked domain lists are mutually exclusive. Schemes, whitespace and hostname wildcards are rejected. Valid path patterns are forwarded unchanged, but Web Fetch does not match path-based filters; use hostnames for effective fetch filtering. No undocumented positive lower bound is imposed on `max_uses` or `max_content_tokens`.

Citations are opt-in and cannot be combined with JSON-schema output, including when citation-enabled fetched documents are replayed. Native fetch successes/errors and streamed inputs remain `ProviderContent`; HTTP 200 with a fetch-result error is not rewritten as a transport failure. `Usage.server_tool_usage.web_fetch_requests` preserves reported zero versus missing values, while `anthropic_usage` and raw provider stream events retain the original usage. A count does not create an inferred fee.

See the [Foundry guide](anthropic-foundry.en.md) for route identity and hosting limits. Validation is offline/mock-based, not live-account acceptance. Sources: [Web Fetch guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool), [tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference), [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry), and [official request schemas](https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/src/resources/messages/messages.ts).

[Server tools: callers, dynamic filtering and domain rules](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools).
