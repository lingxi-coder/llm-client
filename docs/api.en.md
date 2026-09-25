# API Reference

[简体中文](api.md)

This document covers the Rust API in this repository. `lingxi-llm-client` is a library with a built-in HTTP client. It does not provide an HTTP service with a listening port, a CLI, or a key management service. The core client API can be imported from `lingxi_llm_client`; request, response, and configuration types are available through `lingxi_llm_client::protocol`.

Image generation and editing use the independent `client.images()` service; see the [image guide](images.en.md) for requests, tasks, and built-in provider support. `client.chat()` exposes the same conversation operations as the existing `complete()` and `stream()` methods.

## Region filtering

Every client must explicitly select `.with_region(Region::ChinaMainland)` or `.with_region(Region::International)` before `build()`, otherwise it returns `BuildError::MissingRegion`. Import `Region` from `lingxi_llm_client::protocol`; `client.region()` returns the selection. It is fixed for the client lifetime. To switch regions, build another client and reuse the same configuration directory if desired.

`ProviderProfile.regions` declares usage regions, for example `regions = ["china_mainland"]` in TOML. Shared profiles declare `["china_mainland", "international"]`. Missing region declarations in the current format default to both regions; an explicit `[]` permits neither. Models inherit their profile's regions; both `ProviderListing` and `ModelListing` expose `regions`.

`providers()` and `models()` filter by region before applying their existing visibility and model-allowlist rules. Resolution, explicit profile/group references, completions, streams, search and failover chains all honor the region. Excluded models do not cause name ambiguity. Hidden spare accounts remain eligible for failover within the region. This is a product policy, not a network reachability guarantee, IP/language detection or URL rewriting.

`provider()` and `profiles()` retain the complete configuration management view. CRUD, model sync, account usage and standalone file management can still address accounts in other regions. Filtering never deletes their profiles or models, and the client's selected region is not persisted to shared `providers.json`. Current-format profiles without `regions` are available in both regions; restoring a built-in restores its explicit region declaration.

Mainland-only presets: `qwen`, `qwen-search`, `minimax`, `kimi`, `kimi-search`, `glm`, `glm-coding`. `deepseek`, `deepseek-search` and `kimi-code` are shared. All remaining built-in profiles are international, including Qwen Hong Kong, Singapore, US and their search counterparts.


## Contents

- [Getting started and lifecycle](#getting-started-and-lifecycle)
- [Client and builder](#client-and-builder)
- [Requests and messages](#requests-and-messages)
- [Host tool execution and context recovery](#host-tool-execution-and-context-recovery)
- [Web Search API](#web-search-api)
- [Qwen knowledge-base File Search](#qwen-knowledge-base-file-search)
- [Streaming responses](#streaming-responses)
- [Provider configuration and routing](#provider-configuration-and-routing)
- [Authentication and credentials](#authentication-and-credentials)
- [Transport API](#transport-api)
- [Model directory](#model-directory)
- [Usage and cost](#usage-and-cost)
- [Account balances and token usage](#account-balances-and-token-usage)
- [Error handling](#error-handling)
- [Extension APIs](#extension-apis)

## Getting started and lifecycle

1. Use the client within a Tokio async runtime.
2. Load your own `ProviderProfile`, or use `builtin_providers()` / `merge_providers(user)`.
3. Create a builder with `LlmClientBuilder::new(&profiles)?`, select a region with `with_region(Region::International)` or `with_region(Region::ChinaMainland)`, then call `build()`; API key and Bearer authenticators are registered automatically.
4. The host obtains valid credentials and passes them in each request's `RequestOptions`.
5. Call `client.chat().complete()` or `client.chat().stream()`; the host manages conversation history, tool execution, cancellation, and subsequent requests.

See the [README](../README.en.md#1-create-an-application-and-add-dependencies) for dependency setup. The example below uses `serde_json` to construct a configuration, so the calling project also needs to declare `serde_json = "1"` and a Tokio runtime dependency. There is no need to depend directly on `reqwest` or implement transport and a clock yourself.

```rust,no_run
use lingxi_llm_client::protocol::{
    CompletionRequest, ConversationMessage, ProviderProfile, Secret, ToolChoice,
};
use lingxi_llm_client::{LlmClientBuilder, RequestOptions};

async fn ask(api_key: String) -> Result<String, Box<dyn std::error::Error>> {
    let profile: ProviderProfile = serde_json::from_value(serde_json::json!({
        "provider_id": "my-provider",
        "profile_name": "primary",
        "base_url": "https://api.example.com/v1",
        "protocol": "open_ai_chat",
        "auth": "api_key",
        "models": [{
            "display_model": "My Model",
            "request_model": "my-model",
            "billing_model": "my-model",
            "capability_support": {
                "vision": "unknown", "documents": "unknown", "tools": "supported",
                "reasoning": "unknown", "signed_reasoning": "unknown",
                "streaming": "supported", "structured_output": "unknown"
            }
        }]
    }))?;
    let client = LlmClientBuilder::new(&[profile])?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
    let request = CompletionRequest {
        controls: Default::default(),
        service_tier: None,
        model: "my-model".into(),
        web_search: None,
        file_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![ConversationMessage::user_text("你好")],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: Some(1024),
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: serde_json::Value::Null,
    };
    let options = RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..RequestOptions::default()
    };
    let response = client.chat().complete_in("primary", &request, &options).await?;
    Ok(response.message.text())
}
```

`api.example.com` and `my-model` are placeholders; replace them with the service's actual address and model ID. The function does not create an async runtime; call it within a Tokio runtime, for example with `#[tokio::main]`. Tokio can enable the `rt-multi-thread` and `macros` features. For custom transport or tests, use `LlmClientBuilder::with_transport(http, &profiles)`, and optionally call `with_clock(clock)` to set a test clock. Ordinary calls require no service assembly or clock handling.

## Client and builder

### `LlmClientBuilder`

| Method | Return value | Behavior |
| --- | --- | --- |
| `new(&[ProviderProfile])` | `Result<Self, LlmError>` | Creates built-in HTTP transport and a system clock; registers all built-in codecs, directory parsers, and `ApiKey` / `Bearer` authenticators |
| `with_transport(Arc<dyn Transport>, &[ProviderProfile])` | `Self` | Uses custom transport and the built-in system clock; registers all built-in codecs, directory parsers, and `ApiKey` / `Bearer` authenticators |
| `with_region(Region)` | `Self` | Consumes and returns the builder with the required usage region |
| `with_clock(Arc<dyn Clock>)` | `&mut Self` | Overrides the builder's clock, useful for tests with a fixed time |
| `register_codec(Arc<dyn WireCodec>)` | `&mut Self` | Adds or replaces a codec by protocol family |
| `register_directory(Arc<dyn ModelDirectory>)` | `&mut Self` | Adds or replaces a parser by directory protocol shape |
| `register_authenticator(AuthStrategy, Arc<dyn Authenticator>)` | `&mut Self` | Adds or replaces the implementation for an authentication strategy |
| `register_account_source(provider_id, AccountIdentity, Arc<dyn AccountUsageSource>)` | `&mut Self` | Adds or replaces a read-only account source by provider and principal kind |
| `register_profile_account_source(profile_name, AccountIdentity, Arc<dyn AccountUsageSource>)` | `&mut Self` | Binds a signed-in account source to one connection, ahead of provider-wide sources |
| `add_profile(ProviderProfile)` | `&mut Self` | Appends a connection configuration |
| `codec_families()` | `Vec<ProtocolFamily>` | Lists registered protocol families |
| `build(self)` | `Result<LlmClient, BuildError>` | Consumes the builder and validates the configuration |

`BuildError` includes `MissingRegion`, `DuplicateProfile { profile_name }`, `MissingCodec { profile_name, family }`, `MissingAuthenticator { profile_name, strategy }`, and `InvalidPeakSchedule { profile_name, reason }`. `AuthStrategy::None` needs no authenticator; a missing directory parser does not prevent the build. The build rejects invalid or empty peak pricing windows. A successful build does not mean the credentials are valid, the address is reachable, or the provider supports every request parameter.

### `ChatService`

`client.chat()` returns a borrowed `ChatService<'_>` for conversation calls. It uses the same configuration, credentials, routing, attachment preparation, deadlines, and failover as the top-level completion and streaming methods, which remain available. It does not store conversation history or execute tools.

| Method | Return value | Behavior |
| --- | --- | --- |
| `models()` | `Vec<ModelListing>` | List visible models in the current region, filtering out models whose metadata declares `image` output |
| `complete(&CompletionRequest, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | Route by model and return a complete response |
| `complete_in(&str, &CompletionRequest, &RequestOptions).await` | Same | Restrict the starting profile or connection group |
| `stream(&CompletionRequest, &RequestOptions).await` | `Result<ModelStream, LlmError>` | Route by model and open a stream |
| `stream_in(&str, &CompletionRequest, &RequestOptions).await` | Same | Open a stream on the specified profile or group |

For hosted search, set `CompletionRequest.web_search` or `file_search` before calling Chat, or use the `LlmClient::web_search*` convenience methods. `ChatService` has no `web_search*` methods. Configuration, account queries, routing inspection, token estimation, and pricing remain on `LlmClient`; image generation and editing use `client.images()`.

### `LlmClient`

| Service entry | Return value | Purpose |
| --- | --- | --- |
| `chat()` | `ChatService<'_>` | Conversation model listing, completions, and streams |
| `images()` | `ImageService<'_>` | Independent image catalog, generation, editing, and native tasks |

| Method | Return value | Behavior |
| --- | --- | --- |
| `complete(&CompletionRequest, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | Resolves the route, encodes, authenticates, sends, and decodes a complete response |
| `complete_in(&str, &CompletionRequest, &RequestOptions).await` | Same as above | Explicitly selects the starting profile or connection group; model resolution, credentials, and execution use the same route |
| `stream(&CompletionRequest, &RequestOptions).await` | `Result<ModelStream, LlmError>` | Opens an HTTP stream and returns a unified event interface |
| `stream_in(&str, &CompletionRequest, &RequestOptions).await` | Same as above | Opens a stream on the specified profile or connection group |
| `web_search(&CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | Enables provider-hosted search for this request and returns the answer and sources |
| `web_search_stream(&CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<ModelStream, LlmError>` | Enables search for this request and returns streaming events |
| `web_search_in(&str, &CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<CompletionResponse, LlmError>` | Runs search on the specified profile or connection group |
| `web_search_stream_in(&str, &CompletionRequest, WebSearchConfig, &RequestOptions).await` | `Result<ModelStream, LlmError>` | Streams search on the specified profile or connection group |
| `resolve(&str)` | `Result<ResolvedRoute, ResolveError>` | Resolves a model and failover chain from the configuration |
| `resolve_in(&str, Option<&str>)` | Same as above | Explicitly restricts the starting connection or connection group |
| `region()` | `Region` | Returns the usage region selected at construction |
| `models()` | `Vec<ModelListing>` | Returns models from the current effective configuration, including directory-synced models, while skipping connections with `connection.hidden` |
| `providers()` | `Vec<ProviderListing>` | Returns connections in the selected region, including hidden connections and those without credentials |
| `account_usage(&str, &AccountQuery).await` | `Result<AccountSnapshot, AccountUsageError>` | Reads one connection's account limits and usage |
| `accounts_usage(&BTreeMap<String, AccountQuery>).await` | `Vec<(String, Result<AccountSnapshot, AccountUsageError>)>` | Reads every connection with independent results |
| `register_profile_account_source(&str, AccountIdentity, Arc<dyn AccountUsageSource>)` | `Result<(), ProviderStoreError>` | Rebinds a signed-in session after a connection changes |
| `profiles()` | `&[ProviderProfile]` | Reads the configuration used by the client |
| `codec_families()` | `Vec<ProtocolFamily>` | Lists codec protocol families |
| `directory_shapes()` | `Vec<ProtocolFamily>` | Lists protocol shapes supported by directory parsers |
| `directory_for(&ProviderProfile)` | `Option<Arc<dyn ModelDirectory>>` | Looks up a directory parser using `model_list` |
| `estimate_local_tokens(&CompletionRequest)` | `Result<LocalTokenEstimate, LocalTokenCountError>` | Estimates locally visible input using the preferred route and exact model ID |
| `estimate_local_tokens_in(&str, &CompletionRequest)` | Same as above | Estimates input after restricting resolution to a profile or connection group |
| `price_quote(&str, Option<&str>, &PricingContext)` | `Result<PriceQuote, LlmError>` | Queries model/tier rates, sources and matching conditions |
| `estimate_stream_cost(&ResolvedRoute, &ModelStream, Submission)` | `Result<CostEstimate, LlmError>` | Prices a stream using its executed connection, tier and dispatch time |
| `estimate_cost(&ResolvedRoute, &Usage, &PricingContext)` | `Result<CostEstimate, LlmError>` | Estimates cost from configured prices and the current clock |
| `estimate_actual_cost(&ResolvedRoute, &CompletionResponse, Submission)` | Same as above | Estimates cost using the connection that actually succeeded for a complete response |
| `estimate_cost_for_profile(&ResolvedRoute, &str, &UsageReport, &InferenceReport, Submission)` | Same as above | Estimates cost using an executed connection and observed service tier |

The configuration is copied at build time. Local configuration management APIs can update the client's effective configuration; reading data directly through a directory parser does not change the result of `models()`.

## Requests and messages

### `RequestOptions`

| Field | Type / default | Meaning |
| --- | --- | --- |
| `credential` | `Option<Secret<String>>` / `None` | Valid credential for this request; does not read environment variables or static keys from the configuration |
| `fallback_credentials` | `BTreeMap<String, Secret<String>>` / empty | Provides separate credentials by fallback profile name; the first connection's key is not reused if one is missing |
| `total_timeout` | `Option<Duration>` / `None` | Total limit for each request; when omitted, `complete()` defaults to 120 seconds (two hours for video requests) and `stream()` has no total limit |
| `file_account_scope` | `Option<String>` / `None` | Stable, non-secret provider-account identity used to bind and optionally reuse provider file references |

### `CompletionRequest`

| Field | Type | Meaning |
| --- | --- | --- |
| `model` | `String` | Display name, wire ID, alias, or qualified model reference in the configuration |
| `web_search` | `Option<WebSearchConfig>` | `None` disables search by default; `Some` enables hosted search on the selected connection |
| `file_search` | `Option<FileSearchConfig>` | `None` disables it by default; Qwen Responses knowledge-base search configuration |
| `previous_response_id` | `Option<ResponseId>` | Previous response ID for Responses continuation |
| `system` | `Vec<SystemBlock>` | System prompt blocks: `text` and `cacheable` |
| `messages` | `Vec<ConversationMessage>` | Current input and history maintained by the caller |
| `tools` | `Vec<ToolSpec>` | Tool names, descriptions, JSON Schemas, and `strict` flags |
| `tool_choice` | `ToolChoice` | `Auto`, `Any`, `None`, or `Tool { name }` |
| `max_tokens` | `Option<u32>` | Output token limit, mapped by the protocol |
| `temperature` | `Option<f32>` | Sampling temperature; the caller must confirm the provider's supported range |
| `thinking` | `Option<ThinkingConfig>` | Thinking mode, numeric/dynamic budget and effort; see [inference controls](inference.en.md) |
| `service_tier` | `Option<ServiceTier>` | Standard / Fast; omitted preserves the endpoint default |
| `stop_sequences` | `Vec<String>` | Stop sequences |
| `metadata` | `serde_json::Value` | Additional data handled according to the codec implementation; this is not a general promise to pass through arbitrary parameters |

`CompletionRequest` does not implement `Default`. When deserializing with serde, `model` and `messages` are required; other fields have defaults or may be omitted. Protocols express tool choice, thinking, and multimodal content differently. The unified types do not guarantee that every service accepts every combination.

`previous_response_id` applies only to `OpenAiResponses` endpoints configured with `extra.supports_previous_response_id = true`. The caller saves the response ID and sends only the new input needed; the client does not save sessions automatically. Continuation requests do not traverse failover connections because response IDs are endpoint-side state.

### Messages and content blocks

`ConversationMessage { role, content }` has the roles `User`, `Assistant`, and `System`. Convenience methods include `user_text(text)`, `assistant(blocks)`, `text()`, and `tool_uses()`; `text()` concatenates only text blocks, excluding thinking content.

| `ContentBlock` variant | Fields / purpose |
| --- | --- |
| `Text` | `text`, optional `thought_signature`; preserve this field for Gemini replay |
| `ToolUse` | `id: ToolUseId`, `name`, `input: Value`, optional `provider_id` and `thought_signature`; replay real Gemini call IDs and signatures unchanged |
| `ToolResult` | `tool_use_id`, `content`, `is_error`, optional `blocks: Vec<Value>` |
| `Thinking` | `text`, optional `signature`; preserve the signature when replaying |
| `RedactedThinking` | `data`; preserve the raw value returned by the provider |
| `Image` | `source: ImageSource`, supporting Base64 or URL |
| `Document` | `source: DocumentSource` and optional `title`, supporting Base64, text, or URL |
| `Video` | `source: VideoSource`; MiniMax M3 currently accepts an uploaded `video_understanding` file reference |
| `ProviderContent` | `protocol`, `value`; preserves native Responses reasoning, Chat reasoning, and Claude/DeepSeek search content for replay in the next turn |

Base64 sources carry `media_type` and `data`; URL sources carry `url`. The library does not execute tools or automatically download attachments. After a tool call is returned, the host executes the tool and adds its result with the same `tool_use_id` to the next turn's input.

### `CompletionResponse`

`UsageReport` combines `Option<Usage>` with `Missing`, `Partial`, `Complete`, or `Invalid`. Old persisted usage-only objects are rejected. Actual-cost APIs accept only complete reports.

Returns `message: ConversationMessage`, `web_search: Option<WebSearchResult>`, `file_search: Option<FileSearchResult>`, `stop_reason: StopReason`, `usage: UsageReport`, `model: String`, `response_id: Option<ResponseId>`, `inference: InferenceReport`, and `executed_profile: Option<String>`. High-level `complete()` sets the name of the connection that actually succeeded; this field is `None` when decoding directly through a codec. `StopReason` includes `EndTurn`, `ToolUse`, `MaxTokens`, `StopSequence`, `Refusal`, and `Other(String)`.

`inference` preserves requested reasoning effort and service tier, provider-reported tier, and local execution time. Requested values do not confirm the actual tier; see [reasoning and pricing](inference.en.md).

### Host tool execution and context recovery

The following functions live in the calling application. `append_tool_results` takes `response.message` and an application-owned tool executor, preserves the entire assistant message (including signatures and opaque content), and pairs each result with its call ID. The host handles authorization and tool errors in the callback, then chooses whether to send another request. Continue with `response.executed_profile` when present so replay stays on the connection that actually responded.

This example uses full-history replay (`previous_response_id: None`). Stateful continuation instead uses the returned response ID and only new input, scoped to the same connection. The context-recovery helper below illustrates a host policy of one retry: the caller supplies `reduce`, which must preserve valid tool-call/result pairs and replay signatures. It is not an automatic client behavior.

```rust,no_run
use lingxi_llm_client::protocol::{
    CompletionRequest, CompletionResponse, ContentBlock, ConversationMessage,
    LlmError, MessageRole,
};
use lingxi_llm_client::{LlmClient, RequestOptions};

fn append_tool_results(
    request: &mut CompletionRequest,
    assistant: ConversationMessage,
    mut execute: impl FnMut(&str, &serde_json::Value) -> Result<String, String>,
) -> bool {
    let results: Vec<_> = assistant.tool_uses().map(|(id, name, input)| {
        let (content, is_error) = match execute(name, input) {
            Ok(output) => (output, false),
            Err(error) => (error, true),
        };
        ContentBlock::ToolResult {
            tool_use_id: id.clone(), content, is_error, blocks: None,
        }
    }).collect();
    request.messages.push(assistant);
    let has_results = !results.is_empty();
    if has_results {
        request.messages.push(ConversationMessage {
            role: MessageRole::User, content: results,
        });
    }
    has_results
}

async fn call_with_one_context_retry(
    client: &LlmClient,
    profile: &str,
    request: CompletionRequest,
    options: &RequestOptions,
    reduce: impl FnOnce(CompletionRequest, &LlmError) -> CompletionRequest,
) -> Result<CompletionResponse, LlmError> {
    match client.chat().complete_in(profile, &request, options).await {
        Err(error @ (LlmError::ContextOverflow { .. } | LlmError::RequestTooLarge { .. })) => {
            let reduced = reduce(request, &error);
            client.chat().complete_in(profile, &reduced, options).await
        }
        result => result,
    }
}
```

### Migrating from the shared Agent API

The client now defines its own protocol types. Replace old `lingxi-agent-api` imports with `lingxi_llm_client::protocol`; applications keeping separate domain types need their own boundary conversions. There is no Rust type-identity compatibility layer.

- `CompactTrigger` and `triggers_reactive_compaction()` are removed. Keep compaction state in the host and match communication errors as shown above.
- `ProviderProfile::vision_delegate` and the two `MediaDelegation*` errors are removed. Media delegation and its outcomes belong to the host. Remove the field from both configuration and Rust struct literals.
- `OAuthRefreshDead` is removed from `LlmError` and `LlmErrorKind`. Handle refresh failures in the host; an authenticator reporting a request authentication failure returns `Authentication`. Existing authentication strategies and authentication failover remain supported.
- The unused `TokenEstimate` and `TokenEstimateSource` types are removed. Use `estimate_local_tokens()` / `estimate_local_tokens_in()` and their `LocalTokenEstimate` result for local estimates; `Usage` retains provider-reported consumption.

Removed error variants are no longer accepted by the error deserializer. Hosts that persisted those errors should migrate them into their own error model. Provider management, account usage queries, pricing, and local token counting remain available.

## Web Search API

`web_search()` and `web_search_stream()` accept an ordinary `CompletionRequest`, a `WebSearchConfig` for this search, and `RequestOptions`. Both methods only clone the request and set `web_search`, then use the same routing, authentication, and failover logic as `complete()` / `stream()`; the original request is unchanged. The passed configuration overrides any existing `web_search` in the request. Setting `request.web_search = Some(config)` and calling the ordinary methods has the same effect. Here, “API” means Rust client methods; this library does not provide a separate HTTP search service.

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, ConversationMessage, LlmError, Secret, WebSearchConfig};
use lingxi_llm_client::{builtin_providers, LlmClientBuilder, RequestOptions};

async fn search(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let profiles = builtin_providers()?;
    let client = LlmClientBuilder::new(&profiles)?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "glm/glm-4.7",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "查找 Rust 最新版本，给出来源"}]}]
    }))?;
    let options = RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..RequestOptions::default()
    };
    let config = WebSearchConfig::default();
    let response = client.web_search(&request, config, &options).await?;
    println!("{}", response.message.text());
    if let Some(search) = response.web_search {
        for source in search.citations {
            println!("{}: {}", source.title.unwrap_or_default(), source.url);
        }
    }
    Ok(())
}
```

The host supplies `api_key` in the example; the `glm` connection uses an open-platform credential corresponding to `ZHIPU_API_KEY`. The builder does not read environment variables. Other presets that support search include `openai`, `anthropic`, `gemini`, `openrouter`, `zai`, `deepseek-search`, and `kimi-search`. Available models and account permissions are determined by the provider.

The fields of `WebSearchConfig` and the return value are described below; omitting every field lets the provider decide whether to search and what to search.

| Field | Type / default | Effect |
| --- | --- | --- |
| `allowed_domains` | `Vec<String>` / empty | Allow only the specified domains, without `https://` or a path |
| `blocked_domains` | `Vec<String>` / empty | Exclude the specified domains; cannot be set alongside the allowlist |
| `max_uses` | `Option<u32>` / `None` | Maximum searches per request; must be positive; supported only by the Anthropic adapter |

`WebSearchResult.citations` is a list of URLs and optional titles. `metadata` preserves the provider's raw search records, citation positions, errors, and usage. `web_search: None` means the response provided no recognizable search metadata; it does not mean the client confirmed that no search occurred. The server decides whether to call a search tool, so enabling the configuration does not guarantee a search in this turn.

Streaming call:

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, LlmError, StreamEvent, WebSearchConfig};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn search_stream(client: &LlmClient, request: &CompletionRequest, options: &RequestOptions) -> Result<(), LlmError> {
    let mut stream = client.web_search_stream(request, WebSearchConfig::default(), options).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => print!("{text}"),
            StreamEvent::WebSearch { result } => {
                for source in result.citations {
                    eprintln!("source: {}", source.url);
                }
                // 如需准确引用位置或诊断搜索错误，保留 result.metadata。
            }
            StreamEvent::End { stop_reason, .. } => eprintln!("{stop_reason:?}"),
            _ => {}
        }
    }
    Ok(())
}
```

Search events may arrive in multiple frames; the same URL may appear both among sources and in body citations. `WebSearch` is a server metadata event and requires no client tool execution. `ToolChoice::Auto` is the most broadly applicable choice across search connections. See the [complete Web Search documentation](web-search.en.md#support-matrix-and-parameters) for adapter restrictions on filtering, tool choice, and models. A connection without `extra.web_search`, a protocol mismatch, or parameters unsupported by the adapter cause `UnsupportedCapability` before the HTTP request. Invalid domains and nonpositive search counts return `InvalidRequest`. Search tools may be billed separately; `estimate_cost()` estimates only tokens.

## Qwen knowledge-base File Search

Qwen Responses profiles accept one knowledge-base ID and the Model Studio workspace ID. This currently applies to the supported Qwen Max / Flash Responses models; the built-in Beijing, Singapore, US, and Hong Kong Qwen Search profiles declare `extra.file_search = "qwen"`. The client sends File Search requests to the workspace-specific regional domain. Ordinary model requests continue to use the profile's configured base URL.

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, FileSearchConfig, Secret};
use lingxi_llm_client::{LlmClient, RequestOptions};

// Build client with Region::ChinaMainland; international calls need the corresponding profile and credentials.
async fn ask_qwen(client: &LlmClient, api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3.8-max",
        "messages": [{"role":"user","content":[{"type":"text","text":"Answer using the knowledge base"}]}]
    }))?;
    request.file_search = Some(FileSearchConfig {
        knowledge_base_id: "kb-123".into(),
        workspace_id: "ws-example".into(),
    });
    let options = RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..RequestOptions::default()
    };
    let response = client.chat().complete_in("qwen-search", &request, &options).await?;
    if let Some(search) = response.file_search {
        for hit in search.hits {
            println!("{}: {}", hit.filename.unwrap_or_default(), hit.text.unwrap_or_default());
        }
    }
    Ok(())
}
```

`file_search` and `web_search` may be enabled in one request. Streaming calls receive `StreamEvent::FileSearch { result }`; repeated results in the final Responses event are deduplicated. This is Qwen-hosted knowledge-base retrieval, separate from Qwen-Long document uploads and `fileid://` references in the [File Attachments Guide](file-attachments.md).

## Streaming responses

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, LlmError, StreamEvent};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn read_stream(
    client: &LlmClient,
    request: &CompletionRequest,
    options: &RequestOptions,
) -> Result<(), LlmError> {
    let mut stream = client.chat().stream(request, options).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => print!("{text}"),
            StreamEvent::End { stop_reason, .. } => eprintln!("{stop_reason:?}"),
            _ => {}
        }
    }
    if stream.usage_is_complete() {
        if let Some(usage) = stream.observed_usage() {
            eprintln!("tokens: {}", usage.total());
        }
    }
    Ok(())
}
```

`ModelStream` provides its own async `next()`, so there is no need to import `StreamExt`. Use `status()`, `header(name)`, and `headers()` to read the status and response headers from when the stream was opened; header lookup is case insensitive. `executed_profile()` returns the name of the connection that accepted the request and can be used to estimate cost after the stream ends.

| Event | Meaning |
| --- | --- |
| `Start { model, response_id }` | Response starts; the response ID may be absent |
| `TextDelta { block, text }` | Text increment |
| `ReasoningDelta { block, text }` | Thinking increment |
| `ProviderContent { block, protocol, value }` | Complete native reasoning data; retain as `ContentBlock::ProviderContent` at this output index and replay unchanged next turn. Reasoning deltas at the same index are display text and cannot replace this payload |
| `ThoughtSignature { block, signature }` | Thinking signature; preserve it with its corresponding block |
| `RedactedThinking { block, data }` | Opaque thinking data |
| `ToolCallDelta { block, id, name, arguments_fragment }` | Tool argument fragment; accumulate by block before parsing JSON |
| `WebSearch { result }` | Hosted search sources and provider-native metadata; may appear multiple times |
| `FileSearch { result }` | Qwen-hosted knowledge-base retrieval hits; duplicate final output is suppressed |
| `Inference { report }` | Observed service tier and effort, kept separate from requested values |
| `End { stop_reason, usage, inference }` | End event emitted by the decoder |

OpenRouter-style Chat Completions `reasoning` / `reasoning_details` are retained as message-level replay data in `ProviderContent { protocol: OpenAiChat, value: {"type":"chat_reasoning", ...} }`. Buffered responses retain this block; streams emit one complete block before normal termination. Keep one such block in its assistant message, using reasoning deltas only for display. The encoder replays `reasoning_details` in their original order and rejects duplicate envelopes, non-assistant roles, and cross-protocol replay. The existing `reasoning_content` field remains controlled by `preserve_reasoning_content`.

`observed_usage()` may return partial counts. Only `usage_is_complete()` establishes that counts are complete and consistent; the presence of `End` alone does not make them suitable for billing. If the underlying stream reaches EOF before a provider termination marker is observed, the decoder returns `StreamInterrupted`. End the turn after a read error. The client immediately releases the underlying response and pending frames; retaining the handle still permits reading observed usage and response headers. Content errors after `ModelStream` has been returned to the caller do not automatically switch connections, preventing duplicate generation or tool execution.

Anthropic `message_start` counts are initial values. Even a numerically valid `output_tokens` there is not final usage; a later final usage report is needed. Save opaque `RedactedThinking` data by block and put it back unchanged in the next turn's message. Gemini prompt blocking and Responses refusal content retain `StopReason::Refusal`. Complete and streaming responses use the same semantics; truncation and provider errors retain their own terminal states.

Dropping `ModelStream` drops the underlying byte stream; the host's `Transport` should release the corresponding request resources. SSE responses should preserve the correct `Content-Type: text/event-stream`; the library reassembles arbitrary network chunks. The Bedrock decoder reassembles AWS event-stream binary frames itself.

## Provider configuration and routing

### `ProviderProfile`

Required fields are `provider_id`, `profile_name`, `base_url`, `protocol`, and `auth`. `provider_id` is an open string; adding a new vendor does not require a new enum variant. `profile_name` is the unique connection name.

| Field | Meaning / default behavior |
| --- | --- |
| `models` | `Vec<ModelProfile>`, empty by default; a model must first exist in the configuration to be resolved |
| `credential` | Description of the credential source, `None` by default; actual reading is the host's responsibility |
| `model_list` | Omission means the same as `protocol`; `"none"` means unpublished; another protocol family string can also be specified |
| `connection` | Grouping, order, visibility, and failover policy |
| `pricing` | `billingMode` and optional `peak` |
| `signing` | Optional `region`, `service`, and `project`, for the host's signing implementation |
| `azure` | Optional `api_version` and `deployment` |
| `info` | Display name, description, console/API key/documentation links, and credential guidance |
| `extra` | JSON data for protocol options, extra body fields and headers, and so on |
| WebSocket fields such as `supports_websockets` | Configuration metadata; the high-level client currently still uses HTTP |

`ModelProfile` must specify `display_model`, `request_model`, and `billing_model`; it may also include `aliases`, `description`, `metadata`, `capability_support`, `pricing`, and a model-level `billing_mode`. The three model names are used for display/matching, the request protocol, and pricing attribution, respectively. `metadata` holds directory data such as context window, output limit, and modalities.

`capability_support` uses `unknown`, `supported`, and `unsupported`. `ModelProfile::capability_support_for(ModelCapability)` returns explicit support facts; missing fields remain unknown. The old Boolean capability fields and fallback logic have been removed. See `info.features` for inference controls, ranges and combination constraints.

`"capability_support": {"tools": "unsupported"}` only declares tool calling unsupported; other capabilities remain unknown.

`ModelProfile.hidden` defaults to `false`. When set to `true`, `models()` no longer lists the model, but it remains callable explicitly by model name or with `resolve_in()`. At runtime, `set_model_visibility(profile_name, request_model, visible)` can change and save this setting.

### Protocols and URLs

The table shows the paths that this library's encoders actually append. It does not assert that the service is online. `base_url` should not already contain the “appended path” shown in the table.

| `protocol` (serde value) | Codec | `base_url` / appended path |
| --- | --- | --- |
| `open_ai_chat` | `OpenAiChatCodec` | Versioned root path, such as `https://host/v1`; appends `/chat/completions` |
| `open_ai_responses` | `OpenAiResponsesCodec` | Versioned root path; appends `/responses` |
| `anthropic_messages` | `AnthropicMessagesCodec` | Service root path; appends `/v1/messages` |
| `gemini_generate_content` | `GeminiCodec` | Versioned root path; appends `/models/{model}:generateContent` or the streaming action |
| `azure_open_ai` | `AzureOpenAiCodec` | Resource root path; appends `/openai/deployments/{deployment}/chat/completions?api-version=...` |
| `foundry_claude` | `FoundryClaudeCodec` | Same encoding path as Anthropic; appends `/v1/messages` |
| `vertex_claude` | `VertexClaudeCodec` | Root path with project and location; appends `/publishers/anthropic/models/{model}:rawPredict` or `streamRawPredict` |
| `vertex_gemini` | `VertexGeminiCodec` | Root path with project and location; appends `/publishers/google/models/{model}:generateContent` or the streaming action |
| `bedrock_claude` | `BedrockClaudeCodec` | Bedrock runtime root path; appends `/model/{model}/invoke` or `invoke-with-response-stream` |

Azure requires `azure.api_version`; if `azure.deployment` is omitted, `request_model` is used. Hosted-platform codecs handle the URL and wire body; they do not automatically obtain cloud credentials or implement AWS SigV4 signing.

### Resolution and failover

`resolve()` supports full `display_model` names, `request_model` names, aliases, and qualified references of the form `profile/model` or `group/model`. Valid native model IDs containing `/` remain intact. If the same string means both a native ID and a qualified reference to a different target, resolution returns an ambiguity error and sends no request. If a qualifier is both a connection name and a group name, the connection name takes precedence.

When the profile that owns a credential is known, use `complete_in(profile, request, options)`, `stream_in()`, or the corresponding search method. Put the native model ID in `request.model` and pass the profile separately, avoiding concatenation of the target connection and model ID. For example, with both OpenAI and OpenRouter enabled, `complete_in("openai", ...)` with `model: "gpt-4o"` selects OpenAI; `complete_in("openrouter", ...)` with `model: "openai/gpt-4o"` selects OpenRouter. Unqualified `openai/gpt-4o` errors because the two interpretations conflict.

Use `resolve_in(model, Some(name))` to preview the same constrained route, then execute with the corresponding `_in` method. The primary credential must belong to the resolved starting connection; fallback connections can still use only the credentials for their respective profiles in `fallback_credentials`.

The same name across groups returns `ResolveError::AmbiguousAcrossGroups`; a native name and a qualified reference pointing to different targets return `AmbiguousNativeAndQualified`; multiple matching models on one connection return `DuplicateOnProfile`; no match returns `UnknownModel`. Request methods convert resolution errors to `LlmError::ModelUnavailable`.

`ResolvedRoute` includes `provider_id`, `profile_name`, `request_model`, `display_model`, `pricing_model`, `capability_support`, `connection_chain`, and `failover`. Do not treat an arbitrarily hand-constructed route as one already validated by the client.

When `connection.group` is omitted, its group name is `profile_name`. Connections are sorted by `(order, profile_name)`. Fallback connections must be in the same group, offer the same `request_model`, and have the same effective billing mode. A fallback with multiple matching model rows is skipped rather than guessing between row-specific overrides. Qualifying the starting connection does not disable failover within its group. When no connection is specified, visible connections are preferred; an explicitly named profile can still select a hidden connection, and hidden connections can participate in failover.

**Failover is disabled by default.** All fields in `FailoverTriggers::default()` / `NONE` are false; explicitly assigning `FailoverTriggers::DEFAULT` enables all five categories. JSON example:

```json
{
  "connection": {
    "group": "my-provider",
    "order": 0,
    "failover": {
      "rateLimit": true,
      "overloaded": true,
      "serverError": true,
      "network": true,
      "auth": false
    }
  }
}
```

Merge this fragment into a complete `ProviderProfile`. `rateLimit` matches rate limiting and exhausted quotas. `network` matches transport errors and timeouts, including TLS failures that the built-in HTTP client classifies as transport errors. Context overflow and unsupported capabilities do not trigger a switch. The policy comes from the starting route, and each connection is attempted at most once; the library does not wait with backoff or retry the same connection. Once a buffered or streaming request receives a non-success HTTP status, even if reading the error body is interrupted or times out, it retains the status, response headers, and up to 64 KiB of the error body already read; the codec then classifies it to determine failover. An interruption after a successful stream is handed to the caller does not automatically replay the request.

### Built-in configurations and extra parameters

`builtin_providers() -> Result<Vec<ProviderProfile>, PresetError>` returns a static directory compiled into the library. `merge_providers(user)` retains user entries and appends presets that are not overridden by a user entry with the same name. The entire profile is overridden; fields are not merged. The builder still rejects duplicate names in user input. `PresetError` distinguishes parsing failure (`Invalid`) from an empty model list (`NoModels`).

### Local persistence and multiple accounts

After creating an `LlmClient`, call `set_config_dir(path)` to manage `providers.json` in that directory. The library immediately loads its profiles and fixes the directory as an absolute path, so later changes to the working directory do not change where it saves. Account connection settings override the matching static definition. Model fields merge by provenance; see the [v2 migration guide](architecture-migration.en.md). Setting a directory again clears configuration loaded from the previous directory. When several clients write to the same directory, the library coordinates writes with `.providers.json.lock` and reads the latest configuration while holding the lock before applying the current change. External programs that edit the JSON directly must also observe this lock. `sync_provider(profile_name, credential).await` reads the model directory with that connection's own credential, updates existing models by `request_model`, adds new models, and saves the configuration. If the directory is missing, the request fails, or disk writing fails, the old file and in-memory configuration stay unchanged. Sync does not delete old models absent from the directory, or change existing models' `hidden`, pricing, capabilities, or aliases. When the directory explicitly reports an incompatible model, sync persists an exclusion record by profile and excludes the model from effective listings and routing, preventing old configuration, allowlist changes, or a restart from bypassing the exclusion. Full metadata for models still on the allowlist remains persisted and is reused when the directory explicitly restores compatibility. An ID absent from a directory response is not treated as incompatible. A later explicit directory report that the ID is supported, or explicitly replacing, restoring the built-in version of, or deleting the profile, clears the corresponding exclusion records. If a Gemini row lacks `supportedGenerationMethods`, its support state remains unknown; the row is retained, but it does not clear an existing exclusion record.

File reads, file locks, and disk writes during sync run on Tokio's blocking thread pool. `sync_provider()` is the serial convenience entry point. To keep using the client during a directory network request, first call the synchronous `prepare_provider_sync()` to obtain an independent `ProviderSyncOperation`, then run its `fetch().await`, and finally commit with `apply_provider_sync(result).await`. Preparing an operation does not access disk and copies the credential and necessary configuration for that operation. The operation does not borrow the client, so the host can release its own client lock before waiting on the network.

```rust,no_run
use lingxi_llm_client::{LlmClient, ProviderStoreError};
use lingxi_llm_client::protocol::Secret;

async fn refresh(
    client: &mut LlmClient,
    credential: &Secret<String>,
) -> Result<usize, ProviderStoreError> {
    let operation = client.prepare_provider_sync("primary", Some(credential))?;
    // operation 已拥有获取目录所需的数据，可独立调度；不再借用 client。
    let result = operation.fetch().await?;
    client.apply_provider_sync(result).await
}
```

Before fetching the directory, the operation checks the connection configuration on disk; at commit time it validates again and merges the latest configuration while holding the file lock. The result is also bound to the configuration directory generation from preparation; after switching directories, even switching back to the original path, an old result is rejected. Dropping an uncommitted fetch operation does not modify the configuration. The commit checks for cancellation before entering the write, but cancellation after writing starts can still leave an updated file and an outdated client snapshot. If this cancellation occurs, call `set_config_dir()` again to reload persisted state. Synchronous configuration management methods such as `add_provider()` remain blocking interfaces; a host calling them from an async task should schedule them in an appropriate blocking execution environment.

Directory observations have no version numbers. Concurrent fetch results for the same profile are applied in the order their `apply_provider_sync()` calls commit. Freshness validation checks only the connection configuration and configuration directory generation. If the host needs the most recently initiated refresh to take precedence, it should refresh that profile serially or discard stale results before committing.

| Operation | API | Persistence behavior |
| --- | --- | --- |
| Add / modify | `add_provider(profile)` | Adds or replaces the whole profile by `profile_name` and writes to disk; also usable for one account among multiple accounts |
| Query | `provider(profile_name)`, `profiles()`, `providers()`, `deleted_builtin_profiles()` | Reads configuration, listing summaries, and names of soft-deleted built-in entries |
| Delete | `remove_provider(profile_name)` | Removes a custom profile from the file; records a soft deletion for a built-in profile so it remains disabled after restart |
| Restore built-in entry | `restore_builtin(profile_name)` | Clears the soft-deletion marker and saves the library's preset, even if the builder had a custom override with the same name |

Deletion affects one connection account, not other accounts in the same group. To disable an entire provider, delete its profiles one by one. Re-adding a configuration with a soft-deleted name also clears the soft-deletion marker. If the host passes a custom profile to the builder again at every startup, the host must also remove it from its own input.

`set_tracked_models(provider_id, request_model_ids)` sets a global model allowlist for all accounts of the same provider and saves it in `providers.json`. A provider without an allowlist displays or saves no models. Directory sync merges only models in the allowlist, and saving writes only models in the allowlist. Calling `add_provider` before setting the allowlist can still use models supplied in the current session. After restart, expanding the allowlist requires the builder to supply those models again or a new directory sync. If another client changes the model list of a profile with the same name, this client will not restore deleted models from an old cache. `tracked_models(provider_id)` reads the current list; `untrack_model(provider_id, request_model)` removes one model from it, and another sync will not add it back. The allowlist determines which models are tracked, saved, and displayed; model-level `hidden` only determines whether a tracked model appears in `models()`.

Each account for the same provider uses a distinct `profile_name` and `connection.connection_id`, with the same `connection.group`. Set `hidden: false` on the primary account and `hidden: true` on fallback accounts, ordered by `order`. Run `sync_provider` for each account separately; the model directory is not copied between accounts. Failover requires the fallback account's key in `RequestOptions.fallback_credentials`, indexed by its `profile_name`. The configuration file saves only credential-source descriptions such as `env` or `host_managed`, not plaintext keys. Persistence APIs reject `CredentialConfig::Static`.

```rust,no_run
use lingxi_llm_client::protocol::{ProviderProfile, Secret};
use lingxi_llm_client::LlmClientBuilder;

# async fn example(primary: ProviderProfile, spare: ProviderProfile) -> Result<(), Box<dyn std::error::Error>> {
let mut client = LlmClientBuilder::new(&[])?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
client.set_config_dir("./config")?;
client.set_tracked_models("acme", ["model-id".to_owned()])?;
client.add_provider(primary)?;
client.add_provider(spare)?;
client.sync_provider("primary", Some(&Secret::new("key".to_owned()))).await?;
client.set_model_visibility("primary", "model-id", false)?;
assert!(client.provider("primary").is_some());
client.untrack_model("acme", "model-id")?;
client.remove_provider("primary")?;
# Ok(())
# }
```

`ProviderStoreError` distinguishes file, JSON, configuration validation, directory request, pagination, and blocking task errors. Sync reads at most 100 pages and rejects duplicate cursors. If another client modifies the connection configuration during a request, sync returns `ProfileChanged` without writing models from the old connection. To use a saved provider after startup, call `set_config_dir` again; the library does not automatically read credentials from environment variables.

`extra.body` adds JSON fields that the protocol has not written; it cannot override fields already written, model identity, or continuation IDs. `extra.headers` adds non-credential headers; it cannot override headers written by the codec. Leave authentication headers to the authenticator.

Common protocol options include `credential_header` (the header containing the API key), `stream_usage_opt_in`, `preserve_reasoning_content`, `thinking_rejects_forced_tool_choice`, and `supports_previous_response_id`. The respective codecs read these options; they are not guaranteed to affect every protocol.

OpenAI Chat profiles may set `extra.max_tokens_field` to `"max_completion_tokens"` for endpoints that require it; the default is `"max_tokens"`. This is a top-level `extra` option, not an `extra.body` field. Invalid or non-string values return `InvalidRequest`. Chat document URLs return `UnsupportedCapability` before sending; use Base64 content or a protocol supporting URLs.

OpenAI Chat profiles can set `extra.reasoning_tokens_separate = true` when reasoning tokens are reported in addition to completion tokens. The built-in Grok Chat profile enables this; the client normalizes both buffered and streamed output counts to include reasoning. Other profiles keep the default subset convention. Direct codec callers should use `decode_response(response, &context)` and `stream_decoder(&context)` to apply profile-specific conventions.

`extra.body.stream` cannot change the response mode chosen by `complete()` or `stream()`. Known credential headers, including `x-goog-api-key`, and the header named by `extra.credential_header` are rejected when loading or saving `extra.headers`. Pass credentials through request options.

Vertex Claude uses `vertex-2023-10-16` in its request body. Bedrock encodes model IDs/ARNs as one path segment and selects streaming through the path without a body `stream` field.

## Authentication and credentials

The built-in authenticators are stateless unit structs. Both `new()` and `with_transport()` automatically register implementations for `ApiKey` and `Bearer`. They can also be replaced with `register_authenticator(..., Arc::new(...))`:

- `ApiKeyAuthenticator`: Anthropic protocols use `x-api-key` by default, Gemini protocols use `x-goog-api-key`, and OpenAI protocols use `authorization: Bearer ...`. `extra.credential_header` can override the header name, for example `api-key` for an Azure API key.
- `BearerAuthenticator`: Always uses `authorization: Bearer ...`; suitable for a valid token already obtained by the caller.

`AuthStrategy` includes `ApiKey`, `Bearer`, `OAuthBearer`, `CopilotBearer`, `ChatGptOAuth`, `GcpToken`, `AzureToken`, and `None`. A strategy name does not automatically enable login, refresh, or token exchange. Except for the builder-registered `ApiKey` / `Bearer` implementations, the host must register an appropriate implementation for any strategy it uses.

`CredentialConfig::{Env, Static, HostManaged, None}` only describes a source. Built-in authenticators use only `RequestOptions.credential` and return `Authentication` if it is missing. `Secret<String>` redacts Debug/Display output and cannot be serialized. Read the plaintext with `expose_secret()` or `into_inner()`; it does not promise to zero memory. Authenticated `HttpRequest.headers` contain plaintext credentials. The Debug output of `HttpRequest` hides the URL, header values, and body content, but the host still must redact those fields if logging them directly.

`RequestOptions.credential` is used only for the first connection. Automatic failover to a fallback connection requiring authentication needs that profile's credential in `fallback_credentials`; if it is absent, the client returns `Authentication` without sending the fallback request. The host obtains and updates credentials. A custom authenticator can still implement another strategy using the passed `profile`.

## Transport API

### Built-in HTTP client

`HttpTransport::new() -> Result<Self, LlmError>` creates a reusable HTTP client based on reqwest 0.12, Rustls TLS, and streaming responses. Callers need not import reqwest; network requests require a Tokio runtime. `LlmClientBuilder::new(&profiles)?` automatically creates the HTTP client and `SystemClock`, without extra service assembly. For direct model directory calls, `HttpTransport` can be used directly.

| Behavior | Built-in implementation |
| --- | --- |
| HTTPS | Uses Rustls to validate certificates |
| Redirects | Does not follow them by default; `send` preserves the original redirect response |
| Retries | No automatic retry; explicitly configured failover at the high level can still apply |
| Connection timeout | 30 seconds |
| Stream read idle timeout | 60 seconds by default; `HttpTransport::with_read_timeout(Duration)` adjusts it at the client level |
| Total timeout | `complete()` defaults to 120 seconds (two hours for video requests); `stream()` has no total limit by default, so an active long-running stream can continue; `RequestOptions.total_timeout` sets a total limit for either, including stream reads |
| HTTP error status | Preserves status, response headers, and body for codec classification instead of discarding the error body early |
| Network errors | Returns a semantic `LlmError`; its message omits the request URL, authentication headers, and body |

Reusing a client can reuse its connection pool. Dropping a response stream releases its resources. The total limit for a streaming request does not reset when a new chunk arrives. Custom `Transport` can still be injected when different network policies are needed.

### Custom transport and clock

`LlmClientBuilder::with_transport(http, &profiles)` accepts an `Arc<dyn Transport>` and uses the built-in `SystemClock`. `Clock::now() -> SystemTime` is used for price windows; tests can override the system clock with the builder's `with_clock(Arc<dyn Clock>)`. Ordinary usage requires no `Clock` implementation or import.

`Transport: Send + Sync + 'static` uses `async_trait` and requires these methods:

| Method | Return type | Responsibility |
| --- | --- | --- |
| `send(HttpRequest)` | `Result<StreamResponse, LlmError>` | Return raw response bytes, status and headers; never follow redirects or retry automatically |

`HttpRequest` contains `method`, `url`, `headers: Vec<(String, String)>`, `body: Bytes`, and `timeout: Option<Duration>`. `HttpResponse` contains `status: u16`, `headers`, and `body`, and provides case-insensitive `header()`. `StreamResponse` also provides `header()`.

A transport implementation should handle TLS, proxies, connection pooling, timeouts, network error classification, response resource release, and redirect policy. The shared `HttpExecutor` reads at most 64 KiB of the error body of a non-2xx response before passing it to the codec for classification. Reaching the limit, a connection interruption, or an error-body read timeout only truncates the body; it does not discard the status and headers already received. A successful response body must be read completely; an interruption or timeout while reading returns an error. Do not discard an HTTP error status prematurely as a generic network error without a response body.


`HttpExecutor` collects responses, bounds bodies, and enforces monotonic deadlines even when an injected transport ignores timeout. Authentication, uploads, polling and streaming reads consume the same request budget. The codec creates its own SSE or EventStream decoder; `ModelStream` never infers framing from Content-Type.

## Model directory

Built-in directories are `OpenAiChatDirectory`, `AnthropicMessagesDirectory`, and `GeminiDirectory`. If a corresponding reader is absent or `model_list = "none"` is configured, `directory_for()` returns `None`; this does not mean existing models cannot be called. A Responses endpoint using the OpenAI models list should explicitly set `model_list = "open_ai_chat"`.

Directory operations are explicit: construct request → authenticate → transport → decode → fetch the next page using the cursor. The following example applies to a profile given an API key; pass the matching authenticator for other authentication strategies.

```rust,no_run
use lingxi_llm_client::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{Authenticator, ApiKeyAuthenticator, LlmClient, LiveModel, Transport};

async fn list_live_models(
    client: &LlmClient,
    http: &dyn Transport,
    profile: &ProviderProfile,
    key: &Secret<String>,
) -> Result<Vec<LiveModel>, LlmError> {
    let directory = client.directory_for(profile).ok_or_else(|| LlmError::UnsupportedCapability {
        message: "该连接没有可用的模型目录".into(),
    })?;
    let mut models = Vec::new();
    let mut cursor = None;
    loop {
        let mut request = directory.list_request(profile, cursor.as_deref());
        ApiKeyAuthenticator.apply(&mut request, profile, Some(key)).await?;
        let response = lingxi_llm_client::HttpExecutor::new(http).execute(request).await?;
        let page = directory.decode_page(&response)?;
        models.extend(page.models);
        cursor = page.next_cursor;
        if cursor.is_none() {
            return Ok(models);
        }
    }
}
```

A production host should also set an overall refresh time limit and page limit to prevent an abnormal service from repeatedly returning the same cursor. Built-in directory requests have a 30-second timeout per page, enforced by the transport implementation.

The cursor in `ModelPage { models, next_cursor }` is an opaque string; pass it back unchanged. Anthropic's `has_more` must be Boolean. When it is `true`, a nonempty string `last_id` must also be present; missing, incorrectly typed, or contradictory pagination data returns a parsing error. This matches its [model list response structure](https://platform.claude.com/docs/en/api/models/list). Gemini's `nextPageToken` may be omitted, `null`, or an empty string to indicate the final page; other non-string values return a parsing error. If a Gemini row contains `supportedGenerationMethods`, that field must be a string array. Rows that explicitly omit `generateContent` are excluded from callable models; rows explicitly containing it confirm compatibility. If the field is absent, support remains unknown. Such a row may be used for a profile without an existing exclusion record, but does not clear an existing record. `LiveModel` contains `request_model`, optional `display_name`, `description`, `context_window`, and `max_output_tokens`, but no pricing. Reading the directory directly does not automatically add models to the configuration. Use `sync_provider()` or the staged sync interface described above to merge tracked models, or let the host manage the directory itself.

## Usage and cost

`Usage` has mutually exclusive billing buckets: `input_tokens` (uncached input), `output_tokens`, `cache_read_tokens`, and `cache_write_tokens`. `reasoning_tokens` is already included in output tokens and must not be added again. `Usage::total()` sums the four billing buckets. `cache_write_1h_tokens` is the one-hour TTL subset of `cache_write_tokens`, not another addend. Older JSON defaults it to zero; complete Rust struct literals must add it or use `..Usage::default()`. `cache_write_1h_per_million` prices one-hour writes separately; a missing rate with nonzero one-hour usage returns `CostUnavailable`. Legacy aggregate-only cache reports without TTL details continue to use the configured generic cache-write rate. Gemini tool prompt tokens belong to the input bucket, and thought tokens to output.

`Usage.cost: Option<ReportedCost>` is the amount actually reported by the provider. `ReportedCost.nano_usd` is in billionths of a US dollar; `from_usd(f64)` rejects negative, non-finite, and overflowing values. A missing amount means unknown, not free.

`Usage.server_tool_usage` separately preserves provider-reported hosted-tool counts, currently Web Search and File Search requests. These counts are not tokens and are never added to token totals or pricing buckets.

```rust,no_run
use lingxi_llm_client::protocol::{LlmError, PricingContext, Usage};
use lingxi_llm_client::LlmClient;

fn show_estimate(client: &LlmClient, model: &str, usage: &Usage) -> Result<(), LlmError> {
    let route = client.resolve(model)?;
    let cost = client.estimate_cost(&route, usage, &PricingContext::default())?;
    println!("estimated {}: {}", cost.currency, cost.total_cost);
    Ok(())
}
```

Import `CostEstimate` from `lingxi_llm_client::client::pricing`. It includes `pricing_model`, `submission`, five cost fields (input/output/cache_read/cache_write/reasoning), `total_cost`, and the pricing `source`.

- Each `TokenPricing` rate uses its `currency` per million tokens; the default currency is USD. Nonzero usage with a missing corresponding rate returns `CostUnavailable`. Rates in use must be nonnegative and finite; cost calculation overflow also returns `CostUnavailable`.
- Missing prices or essential billing conditions return `CostUnavailable`; use `price_quote` to query price availability.
- `Submission::Batch` uses explicit batch rates for each bucket. It does not assume a fixed discount or mean that the client submits batch jobs.
- If reasoning has a separately configured price, reasoning tokens are taken out of the output bucket to avoid double billing; `reasoning_tokens > output_tokens` returns `CostUnavailable`.
- `PeakSchedule` uses UTC windows (`HH:MM-HH:MM`, with `24:00` allowed as an end time) and optional weekday constraints. Set `PricingContext.unix_seconds` to quote a specific time. All public estimates share the price-rule selector; `TokenPricing::at` and the public `pricing::estimate` entry point have been removed.

Cost is a directory estimate, not a replacement for the provider's bill. `estimate_cost` is only for a pre-request estimate of the starting route. After failover, use `estimate_actual_cost(&route, &response, submission)`. For streams, use `estimate_stream_cost(&route, &stream, submission)` to include the executed connection and actual service tier. Both methods that price the actual connection still use directory prices, which may differ from the provider's bill.

### Local input token estimates

```rust,no_run
use lingxi_llm_client::{
    protocol::CompletionRequest, LocalTokenCountError, LocalTokenEstimate, LlmClient,
};

fn estimate_input(
    client: &LlmClient,
    request: &CompletionRequest,
) -> Result<LocalTokenEstimate, LocalTokenCountError> {
    let estimate = client.estimate_local_tokens(request)?;
    println!("{} tokens via {}", estimate.input_tokens, estimate.tokenizer);
    if estimate.is_partial {
        println!("not counted: {:?}", estimate.uncounted_components);
    }
    Ok(estimate)
}
```

Both methods are synchronous. They make no network requests, read no credentials, resolve no remote URLs, and call no attachment resolver. `estimate_local_tokens` resolves the model in the client's selected region and uses the preferred route. `estimate_local_tokens_in(profile_or_group, request)` restricts resolution to a connection or connection group. Only the preferred route's model is counted; later failover connections are not predicted.

The estimator counts input only: system and conversation text, text documents, tool names/descriptions/JSON schemas, tool calls, and text tool results. Structured input is serialized as compact JSON; message boundaries, tool wrappers, and provider request formatting use fixed overhead estimates. `max_tokens` is an output limit and is not added to input tokens. Generation settings such as temperature are not prompt text. `LocalTokenEstimate.is_estimate` is always `true` because provider message framing and hidden templates can differ. Treat this as a local estimate; provider-reported `Usage` remains authoritative for actual consumption.

Default builds contain no tokenizer backend or embedded tokenizer assets. Enable `tokenizer-openai`, `tokenizer-deepseek`, `tokenizer-qwen`, `tokenizer-kimi`, `tokenizer-glm`, or the aggregate `tokenizers-all`. A known model whose backend is disabled returns `FeatureDisabled`; an unknown model returns `UnsupportedModel`.

Local tokenizers are selected only when a bundled asset is explicitly matched to a model: OpenAI uses `tiktoken-rs` model mappings; DeepSeek `deepseek-v4-pro` and `deepseek-flash`; Qwen `qwen3.8-flash` and `qwen3.8-max`; Kimi `kimi-k3`; and Z.AI/Zhipu `glm-5`. Both the provider and model ID must match. Other models, Anthropic, Gemini, xAI, OpenRouter, and MiniMax M3 return `LocalTokenCountError::UnsupportedModel`. Official token counting for Anthropic, Gemini, and xAI uses online endpoints; OpenRouter usage comes from upstream providers. Unknown model IDs never fall back to character-ratio estimates.

When input is unavailable locally or cannot be counted as text, the method still returns an estimate for visible text, sets `is_partial = true`, and lists every omission in `uncounted_components`: images, non-text or remote documents, video, provider files, hosted Web Search/File Search context, previous Responses server state, provider-specific structured blocks, signatures, and provider metadata. Encrypted thinking is never tokenized as plaintext: Anthropic and its hosted variants report it as an opaque-content omission; other built-in protocols skip this block because they do not send it. Text in text documents is counted. Remote resources are not downloaded and attachments are not read. Tokenizer mappings, sources, licenses, and SHA-256 hashes are recorded in [`data/tokenizers/README.md`](../data/tokenizers/README.md). The source archive includes XZ-compressed assets and licenses; each enabled feature embeds only its corresponding assets. The MiniMax M3 asset is excluded because its upstream license restricts use to non-commercial purposes.

## Account balances and token usage

`AccountQuery.execution` defaults to a 60-second budget per account and 30 seconds per HTTP/RPC operation. The total budget starts when a batch slot is obtained and does not restart for pages or subsequent calls. Batch concurrency defaults to 4 and is adjustable with `with_account_concurrency(NonZeroUsize)`. Completed queries free slots immediately, while returned results retain profile order. `AccountFailure::Timeout` marks only unfinished fields; already committed metrics are preserved. Custom sources implement `fetch(&AccountFetchContext, &mut AccountReport)` and write each completed field before awaiting the next operation.

`LlmClient::account_usage(profile_name, &AccountQuery)` reads one connection's account status. `accounts_usage(&BTreeMap<String, AccountQuery>)` reads every configured connection, including hidden ones, and returns an independent result per connection. A missing query yields `AccountUsageError::MissingQuery` only for that profile. `AccountQuery::new(AccountIdentity::ApiKey | AccountIdentity::AuthUser)` identifies the actual principal explicitly: a normal API key can use a Bearer header, so `ProviderProfile.auth` alone does not distinguish account types. The default history range is the past 30 days in UTC Unix seconds; `since_unix` and `until_unix` override it.

```rust,no_run
use lingxi_llm_client::{AccountIdentity, AccountMetric, AccountQuery, LlmClient};
use lingxi_llm_client::protocol::Secret;

async fn show_balance(client: &LlmClient, key: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new(key));
    let account = client.account_usage("deepseek", &query).await?;
    if let AccountMetric::Available { scope, value, .. } = account.balance {
        println!("scope: {:?}, balances: {:?}", scope, value);
    }
    Ok(())
}
```

Each `AccountSnapshot` field (`balance`, `token_usage`, `cost_usage`, `quota_windows`, `subscription`) has independent availability: available, undocumented, missing credentials, not reported, or failed. Available data carries its actual key, user, project, team or organization scope. Organization totals are never presented as exclusive to one connection. `AccountTokenBucket.input_tokens` includes cached input when the provider reports it; `cached_input_tokens` and `cache_write_tokens` are subsets. These account statistics do not use the per-request `Usage` billing-bucket convention. Balances and historical costs are returned as decimal strings; cost amounts use major currency units. Quota windows retain provider-reported durations, percentages, reset times and absolute counts; monetary limits use `limit_decimal` and `remaining_decimal`. Absent details remain `None`. Only official entitlement evidence can set `VerifiedActive` or `VerifiedInactive`.

Ordinary keys can query DeepSeek and Moonshot balances and OpenRouter key spending caps. DeepSeek's `AccountBalance.is_available` preserves whether the provider says a reported balance can be spent. A key cap is not an account balance; OpenRouter account credits need a separate Management Key. OpenAI, Anthropic and xAI account endpoints require Admin/Management credentials; Google project usage requires Cloud Monitoring permission and a project ID. Pass privileged credentials in `management_credential` and required `api_key_id`, `team_id`, `project_id` and similar identifiers in `AccountQuery.selector`. Management requests use fixed official origins, never a model connection's custom `base_url`. An undocumented field returns `Unsupported`; a model response or catalog price is not treated as an account bill.

Qwen quota windows require a regional Qwen API key and `selector.workspace_id`. `model_limit` / `workspace_limit` are rate or usage caps, not consumed counts, so `used`, `remaining`, and percentages stay absent. Qwen cost history uses a separate signed GetBillingTrend request with `AlibabaAccessKey { id, secret, security_token }`; `selector.api_key_id` is also required. Results are daily model costs for the selected region. AccessKey credentials live only in that `AccountQuery`; the client never persists or logs them.

```rust
use lingxi_llm_client::{AccountIdentity, AccountQuery, AlibabaAccessKey};
use lingxi_llm_client::protocol::Secret;

let mut query = AccountQuery::new(AccountIdentity::ApiKey);
query.credential = Some(Secret::new("qwen-api-key".into()));
query.selector.workspace_id = Some("ws-example".into());
query.selector.api_key_id = Some("key-id-from-aliyun".into());
query.alibaba_access_key = Some(AlibabaAccessKey {
    id: "ram-access-key-id".into(),
    secret: Secret::new("ram-access-key-secret".into()),
    security_token: None,
});
```

MiniMax Token Plan quota windows use `AccountQuery.credential` to query `/v1/token_plan/remains`. Ambiguous `current_interval_usage_count` / `current_weekly_usage_count` values are not interpreted as either consumed or remaining usage; only explicit remaining counts or percentages are mapped. Pay-as-you-go balance and billing history are not reported as supported.

For ChatGPT/Codex, GitHub Copilot and Kimi Code user accounts, the host supplies a signed-in official local service or SDK. One account can use `register_account_source(provider_id, AccountIdentity::AuthUser, source)`; for several Auth users of the same provider, bind each signed-in session with `register_profile_account_source(profile_name, AccountIdentity::AuthUser, source)`. A shared single-session source returns `AmbiguousAccountSource` rather than attributing one user's quota to another. When supplied, `AccountQuery.credential` is passed to Copilot `account.getQuota` as the user's `gitHubToken`. `CodexAccountSource` and `CopilotAccountSource` accept a host-implemented `AccountRpc`; its `call` returns the JSON-RPC `result` value and maps RPC errors to `AccountFailure`. `KimiCodeAccountSource` uses a loopback service URL and token provided per query. The client does not perform OAuth login, refresh or credential persistence. Codex five-hour/weekly and Kimi Code five-hour/available seven-day windows are returned when present; Copilot returns only windows reported by its SDK. Codex daily token records are whole UTC-day buckets: the requested range selects every overlapping full day without prorating by hour. Kimi Code's documented `userinfo` gives a level name but no unambiguous subscription-state contract, so its subscription status remains `Unknown`. GLM Coding Plan has no confirmed public account-query API and is not marked subscribed from profile configuration alone.

Copilot can also receive the queried user's GitHub token in `AccountQuery.credential`; then one SDK source can query several users without profile-specific binding.

Replacing, removing, or reloading a changed connection clears its profile-bound account source; switching configuration directories clears those bindings as well. The implicit binding of a provider-wide single-account session source also expires: subsequent queries requiring a session binding return `AmbiguousAccountSource`. Stateless sources and queries that explicitly supply the user token remain reusable. After signing in again, call `LlmClient::register_profile_account_source` so an old session cannot be attributed to the new connection.

## Error handling

`LlmError` is defined in the client's `protocol` module; `kind()` returns an `LlmErrorKind` suitable for classification tables. Do not control flow by matching `message` text.

| Variant | Meaning / host response |
| --- | --- |
| `Authentication`, `PermissionDenied` | Check credentials, refresh, or permissions |
| `InvalidRequest`, `UnsupportedCapability` | Adjust the request or configuration |
| `RateLimited { retry_after, .. }`, `QuotaExceeded` | Rate limit/quota; back off or show the limit |
| `ContextOverflow { limit, actual, .. }`, `RequestTooLarge` | Reduce history or request body |
| `ModelUnavailable` | Model cannot be resolved or is unavailable |
| `ProviderInternal`, `Overloaded` | Provider internal failure/overload |
| `Transport`, `TransportTimeout`, `TlsCert` | Network, timeout, or TLS problem |
| `StreamInterrupted` | Stream content corrupted or unexpectedly interrupted |
| `ProviderFileProcessing { file, .. }` | File readiness is unresolved; retain the account-scoped reference to resume polling later |
| `CostUnavailable` | Insufficient pricing data |

Except for the additional fields shown, these variants all contain `message: String`. `retry_after` is an optional `Duration`; built-in parsers currently support `Retry-After` in seconds. Hosts match `ContextOverflow` and `RequestTooLarge` directly and decide whether to reduce input and call again; the client does not automatically compact or retry these errors.

## Extension APIs

The optional `directory::DecodedModelPage` adds compatibility metadata: `page: ModelPage`, `incompatible_model_ids`, and `explicitly_compatible_model_ids`. `LiveModel.inference_features` carries explicit directory observations of inference controls. `ModelDirectory::decode_page_with_exclusions()` calls the existing `decode_page()` by default and returns two empty ID sets.

| Interface | Required methods | Registration / use |
| --- | --- | --- |
| `WireCodec` | `family`, `encode_request`, `encoded_body_len`, `decode_response`, `stream_decoder` | Builder's `register_codec`; later registration replaces an earlier codec for the same protocol family |
| `StreamDecoder` | `push_bytes`, `finish`, `usage_report` | Codec creates a separate stateful decoder for each response |
| `Authenticator` | Async `apply(&mut HttpRequest, &ProviderProfile, Option<&Secret<String>>)` | Builder registers by authentication strategy; called after encoding |
| `ModelDirectory` | `shape`, `list_request`, `decode_page`; optional `decode_page_with_exclusions -> directory::DecodedModelPage` | Builder registers by directory shape; `DecodedModelPage` contains the original `ModelPage`, explicitly incompatible IDs, and explicitly compatible IDs; the default method preserves old decoding behavior and returns empty sets |

`ProtocolFamily` is a closed enum. A new provider using an existing compatible protocol only needs configuration; a new protocol family requires changes to the enum and a codec. A codec should turn non-success responses into semantic `LlmError` values, normalize token usage, and preserve thinking signatures and tool IDs.

`framing::sse::SseFrameSplitter` provides `new()`, `push(&[u8])`, and `finish()`; it returns an error when an incomplete event exceeds 8 MiB. `framing::eventstream` parses AWS binary frames and limits the size of each frame. These are normally used directly only for custom transport/codec integrations.

Source index: [client](../src/client/mod.rs), [shared requests/responses](../src/protocol/llm.rs), [configuration](../src/protocol/provider.rs), [transport](../src/transport.rs), [integration tests](../tests/). Generate item-by-item Rust API documentation locally with:

```sh
cargo doc --no-deps --open
```


See [inference controls and service-tier pricing](inference.en.md) for `info.features`, `info.pricing`, `price_quote`, `estimate_cost`, and `estimate_stream_cost`.

## Host-owned retries and accounting

Applications that own admission, cancellation and durable accounting can call
`prepare_on(profile, request, options, mode)`. The profile must be an exact
connection name, not a group. Preparation pins the selected model row, encodes
and authenticates the request, and may upload attachments; it does not send a
generation request. `PreparedCall` cannot be cloned. Inspect `request()` and
capture `pricing_snapshot()` before consuming it with `dispatch_once()`.
Every retry or failover requires a new prepared call.

Inspect `ReceivedCall::status()` before choosing `into_stream()` or `collect()`.
The collected usage and inference reports are available before `decode()`, even
on unsuccessful HTTP responses or malformed answer content. Call `finish()` to
complete attachment cleanup. `ModelStream::next_batch()` exposes observations
from each received transport chunk, including usage-only chunks, without
waiting for another chunk after decoding. Retain these observations before
yielding control or validating an application's output schema.

`count_tokens_exact_in` uses the Anthropic counting endpoint. Unsupported wires
return `None`; errors remain errors. Counting never generates an answer.
`FrozenPricing::estimate` uses the selected prices and confirmed execution
facts, retains currency, and rejects partial usage or an unknown Fast tier.
It does not own an application ledger.

For Responses WebSocket, inject a `Transport` implementing `connect_websocket`.
Establish the connection before admission with `PreparedCall::connect_websocket`,
then consume the prepared call with `dispatch_websocket_once`. This sends one
`response.create`; it never reconnects or falls back. The `websocket` module
provides continuation-state helpers. The host owns connection lifetime and
must settle an attempted send before starting a fallback attempt. The default
HTTP transport does not provide WebSocket connections.

`CompletionRequest::controls` carries sampling and structured-output controls,
Anthropic context hints, and Responses storage, continuation and metadata
options. System cache controls preserve TTL and scope. Native content is
protocol-tagged and cannot be replayed on another wire. Streams expose native
annotation deltas and block-end events; these are not application tool calls.

### Host finalization and reusable sessions

Use `prepare_draft_on` when host policy must modify a request before signing.
`RequestDraft` cannot dispatch. Finish changes through `request_mut()`, then call
`seal()` to authenticate the final bytes and obtain an immutable, non-cloneable
`PreparedCall`. Drafts use the explicit `total_timeout`, with no implicit total
deadline; `prepare_on` and ordinary completion calls keep their default timeout
behavior. `RequestOptions::finalizer` applies a synchronous post-encoding,
pre-authentication transformation. `exact_json::serialize` preserves exact UTF-16
code units, including lone surrogates. `auth::sigv4` contains the pure AWS signer;
the host still owns credential acquisition and refresh.

`dispatch_once_with` and `dispatch_websocket_once_with` run a synchronous hook
immediately before sending. Rejection sends nothing; no credential refresh or
retry happens after the hook. `dispatch_once_using` supports a borrowed platform
transport while returned streams independently own their read resources.

`ResponsesSession` owns connection reuse, prewarming, continuation and response
observation. Call `prepare` to connect and compute the incremental request, then
`seal`, perform host admission and `dispatch`. A handshake 426 may select HTTP
before any generation is sent. A send-stage failure only affects the next
attempt: it never resends the current call. Cancelling an unfinished response
invalidates continuation and requires a new connection.

`FileService::upload_gemini_unpolled` and `poll_gemini_active` expose separate
upload and readiness phases. Upload URLs must share the configured origin; the
second leg uses the temporary upload capability without repeating the API key.
`RoutingCatalog` resolves models without networking, with an explicit
`resolve_in_prefer_native` policy for applications whose native model IDs take
precedence over UI-qualified names. `FrozenPricing::capture`,
`with_token_pricing` and `quote` support captured declarations and admission
quotes; actual estimates still require complete usage and execution facts via
`estimate`.
