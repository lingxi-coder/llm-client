# Reasoning controls, capabilities and service-tier pricing

[中文](inference.md)

Effort changes how much reasoning a model performs and can change token consumption. It is not a unit-price dimension. Rates are selected by model, standard/fast service tier and published context, submission or time conditions. Fast multipliers apply only to the model and token buckets explicitly declared in the price rule.

The [complete compilable example](../examples/inference_pricing.rs) queries metadata, chooses controls, sends a request and reads charges. Running it sends one billable request using `OPENAI_API_KEY`.

## Discover capabilities and prices

`providers().info.features` describes the connection and protocol. `models().info.features` describes the effective capabilities of a model on that connection. `Supported / Unsupported / Unknown` distinguishes explicit facts from missing information. Budget ranges, dynamic budgets, thinking modes, accepted efforts, defaults, access requirements and documentation sources are exposed. API support does not guarantee account access to a preview.

`mode_support` records partial mode observations and mode-specific effort/tool restrictions; `effort.level_support` records individual effort observations. Use `features.supports_mode(mode)`, `features.supports_effort(Some(mode), effort)` and `features.forced_tool_choice(mode)` to apply these restrictions consistently with request validation. Missing directory fields preserve previous facts. Anthropic Models API observations include these partial facts; OpenRouter's explicit `supported_efforts: null` replaces a previous effort list with all accepted gateway efforts.

`effort.with_disabled_thinking` states whether a non-`none` effort can accompany explicitly disabled thinking. It includes the configured wire's constraints: OpenAI rejects that combination, while Anthropic permits it subject to model restrictions. Explicitly enabled or adaptive thinking cannot accompany `effort: none`. These combination checks are shared by the capability helper and request validation; omitting an effort preserves the provider's behavior for the selected mode. Clearing model overrides retains the connection and wire constraints.

For example, Claude Opus 5 accepts disabled thinking only with low/medium/high effort. Fable 5 and Fable 5.1 cannot disable thinking. Manual thinking disallows forced tool choice; adaptive thinking supports it subject to model restrictions, including Fable 5.1. These rules are exposed in the same capability data that validates requests. [Claude thinking reference](https://platform.claude.com/docs/en/build-with-claude/thinking).

`ModelProfile.pricing` is the canonical configuration; runtime `info.pricing` is its read-only projection. Edit prices with `ModelField::Pricing`, and capabilities with `ModelField::InferenceFeatures`. Directory observations replace only explicitly reported facts. Missing fields preserve catalog facts; user overrides take precedence.

```rust,no_run
use lingxi_llm_client::{LlmClient, protocol::*};

fn inspect(client: &LlmClient) -> Result<(), LlmError> {
    for model in client.models() {
        println!("{}: {:?}", model.request_model, model.info.features);
    }
    let quote = client.price_quote("claude-opus-5", Some("anthropic"),
        &PricingContext {
            service_tier: Some(ServiceTier::Fast),
            input_tokens: Some(2000),
            ..Default::default()
        })?;
    println!("{:?} {} {:?}", quote.status, quote.currency, quote.rates);
    println!("multiplier: {:?}; source: {:?}", quote.multiplier, quote.source);
    Ok(())
}
```

`PriceQuote` returns `Priced / Unknown / Unsupported`. Individual rates remain optional: estimation fails when a consumed bucket has no published price. Context bands need the full prompt length, including cached input. Rates use the quoted currency per million tokens: MiniMax mainland uses CNY and its international service uses USD, without implicit FX conversion. Published subscription consumption multipliers appear in `quota`; they are neither dollar prices nor free tokens.

Rule `valid_from` (inclusive) and `valid_until` (exclusive) store `PriceBoundary { local, time_zone }`, for example `local = "2027-01-01T00:00:00"` with `time_zone = "America/Los_Angeles"`. IANA rules account for DST before comparison with `PricingContext.unix_seconds`; the machine time zone is never used. Ambiguous or nonexistent local times require a valid explicit RFC 3339 offset, and invalid zones fail configuration validation. If the official zone and offset are unpublished, prices remain unknown during the possible UTC−12 to UTC+14 transition window. Quotes retain the source zone, matched rule and verification date. An omitted tier in `PricingContext` assumes standard pricing.

## Configure requests

```rust,no_run
use lingxi_llm_client::protocol::*;

fn request() -> CompletionRequest {
    let mut request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-opus-5",
        "messages": [{"role":"user", "content":[{"type":"text", "text":"Explain this design."}]}],
        "max_tokens": 8192
    })).unwrap();
    request.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Adaptive),
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    });
    request.service_tier = Some(ServiceTier::Fast);
    request
}
```

Use `ThinkingBudget::Tokens(n)` for a numeric budget, `ThinkingBudget::Dynamic` for dynamic allocation, and `ThinkingMode::Disabled` to turn thinking off. The model must support each setting. `max_tokens` remains a separate output limit; a thinking budget is generally a target rather than a guaranteed spend.

| Connection | Mapping |
| --- | --- |
| OpenAI | Chat `reasoning_effort`, Responses `reasoning.effort`; fast tier |
| Anthropic | Model-specific manual budgets or adaptive/effort; fast speed and beta header |
| Gemini | `thinkingBudget` (-1 dynamic, 0 disabled), `thinkingLevel`; `serviceTier: "priority"` |
| Qwen | Chat `enable_thinking/thinking_budget`; Responses effort |
| DeepSeek | Thinking and effort; numeric budgets that would be ignored are rejected |
| Kimi, GLM/Z.AI | Model- and protocol-declared thinking and effort |
| MiniMax | Model-declared thinking; priority admission |
| Grok | Model-declared effort; Chat/Responses priority |
| OpenRouter | `reasoning` object; priority separate from throughput routing |
| Copilot | Unverified direct HTTP controls remain unknown; product UI support is not an API contract |

Highspeed/Turbo variants are explicitly selected models. The client never swaps them automatically. Known unsupported controls produce `UnsupportedCapability`; invalid budgets, contradictory settings and `extra.body` conflicts produce `InvalidRequest`. Typed controls and body extras cannot assign different values to the same field. Every failover attempt preserves and validates the settings. Custom codecs can implement `WireCodec::validate_request` to validate before uploads.

## Actual service tier and charges

The response's `inference`, the stream's `inference_report()`, `StreamEvent::Inference` and terminal event retain requested effort/tier, provider observations and raw strings. Requested values are not confirmations. Gemini `usageMetadata.serviceTier`, response headers and other protocols' JSON/SSE observations are preserved. Native Gemini `serviceTier` and its protobuf `service_tier` spelling are checked for conflicts and emitted once as `serviceTier`. Missing observations remain unknown.

```rust,no_run
use lingxi_llm_client::{LlmClient, ResolvedRoute, ModelStream, protocol::*};

fn report(client: &LlmClient, route: &ResolvedRoute, response: &CompletionResponse,
          stream: &ModelStream) -> Result<(), LlmError> {
    if let Some(cost) = response.usage.usage.and_then(|usage| usage.cost) {
        println!("provider reported nano-USD: {}", cost.nano_usd);
    }
    let cost = client.estimate_actual_cost(route, response, Submission::Interactive)?;
    println!("token estimate: {cost:?}; inference: {:?}", response.inference);
    let streamed_cost = client.estimate_stream_cost(route, stream, Submission::Interactive)?;
    println!("stream token estimate: {streamed_cost:?}");
    Ok(())
}
```

Use `estimate_cost` for preflight estimates at a chosen tier. When `PricingContext.service_tier` is omitted, quotes and estimates use the declared model or connection default, including Fast; an explicit tier takes precedence. `estimate_actual_cost` and `estimate_stream_cost` use the executed connection, complete usage and observed tier. A fast request or a configured fast default with an unknown actual tier, or an unpriced tier, produces `CostUnavailable`, never a standard-price fallback. OpenAI Responses cache writes are removed from ordinary input and billed at the cache-write rate; reads plus writes must fit within total input. Reasoning is counted once as a subset of output. One-hour cache writes are separated from total cache writes and priced at their own rate. Estimates cover token charges, excluding unrepresented hosted-tool and cache-storage charges; provider-reported money remains separate.

Pricing uses the successful attempt’s local dispatch time, `inference.executed_at`, so later queries cannot move it into a different price period. It is not a provider-confirmed timestamp. Time-dependent prices without an execution time produce `CostUnavailable`, including when a fast multiplier inherits dated standard prices.

## API migration

- Replace `ThinkingConfig::budget_tokens` with `budget: Option<ThinkingBudget>`; `mode/effort` and request `service_tier` are new.
- Responses and terminal events include `inference`; exhaustive stream matches must handle the non-text `Inference` event.
- `CostEstimate` exposes `currency` and `input_cost/output_cost/cache_read_cost/cache_write_cost/reasoning_cost/total_cost`, instead of labeling every currency USD.
- `estimate_cost` requires `PricingContext` and returns `CostEstimate`; unknown prices return `CostUnavailable`. `estimate_cost_for_profile` requires `InferenceReport`.
- Configuration uses v2 explicit overrides, without old field aliases, automatic migrations or API wrappers. Numeric budget JSON must use `{"budget":{"tokens":2048}}`; `budget_tokens` and bare integers are rejected.
- Boolean `capabilities` are removed. Use tri-state `capability_support` for model, listing and resolved-route metadata.

- `TokenPricing::at` and public `pricing::estimate` are removed. Use the client quote/estimate methods with `PricingContext`; all of them select the same context, tier, submission and time rules.
