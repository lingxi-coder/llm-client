# llm-client

[简体中文](README.md)

`llm-client` is a standalone Rust library for calling different LLM services from an application. Applications use shared request and response types and choose models and connections through provider profiles; the client handles protocol encoding, HTTP transport, stream parsing, and error classification. It also supports model catalogs, provider-hosted Web Search, usage tracking, and cost estimation.

Protocol types are defined by this crate and available through `lingxi_llm_client::protocol`. Applications own tool execution, permissions, conversation history, credential refresh, and context compaction; see the [API guide](docs/api.en.md#host-tool-execution-and-context-recovery).

## Features

- [Reasoning controls and fast pricing](docs/inference.en.md): discover model capabilities, configure budgets/effort/fast, and quote standard or fast rates. Effort changes consumption, not unit prices.
- **Multiple protocols, one API**: built-in codecs for OpenAI Responses, Chat Completions, Anthropic Messages, and Gemini, plus hosted adapters for platforms such as Azure, Bedrock, and Vertex, reduce protocol-specific application code.
- **Configuration-driven integration**: built-in provider and model profiles; add services compatible with existing protocols through configuration. Configure multiple accounts for one provider, sync their catalogs separately, and control model visibility.
- **Routing and failover**: resolve connections by model or select a profile explicitly. Configure optional backup connections and credentials, and use response metadata to identify the connection that handled the request.
- **Consistent streams and search results**: handle text, tool calls, and usage through shared stream events. Access provider-hosted search and citations through the Web Search API, with validation of supported search options and model capabilities.
- **Independent image service**: generate and edit images or query native tasks through `client.images()`, with image requests and capabilities separate from Chat. See the [image guide](docs/images.en.md).
- **Usage and cost tracking**: normalized input, output, and cache token usage, with cost estimates based on catalog prices and the connection that actually served the request.
- **Offline input token estimates**: count visible request content with optional provider-specific tokenizers, including system messages, tools, and conversation text; report uncounted components explicitly.
- **Account limits**: read documented provider balances, historical token usage, quota windows, and coding-plan entitlements per connection, with explicit account identity and data scope.
- **Cross-device file attachments**: conversations keep stable application-owned references; each model request resolves them to inline content or a supported provider file input for its active connection. Display and model-input lifetimes stay separate.
- **Built-in defaults, extensible interfaces**: HTTP transport and authenticators are included; replace transport or clocks, or add protocols. Applications supply credentials per request; secrets are not persisted in provider configuration.

### Supported LLM Providers

The repository includes these connection presets in [`data/providers/`](data/providers/). Each name in the table is a profile name accepted by methods such as `complete_in()`.

| Service | Built-in profiles |
| --- | --- |
| OpenAI | `openai` |
| Anthropic | `anthropic` |
| Google Gemini | `gemini` |
| DeepSeek | `deepseek`, `deepseek-search` |
| Kimi | `kimi`, `kimi-intl`, `kimi-code`, `kimi-search`, `kimi-search-intl` |
| Qwen / Model Studio | `qwen`, `qwen-intl`, `qwen-us`, `qwen-hk`, `qwen-search`, `qwen-search-intl`, `qwen-search-us`, `qwen-search-hk` |
| MiniMax | `minimax`, `minimax-intl` |
| Z.AI / GLM | `zai`, `zai-coding`, `glm`, `glm-coding` |
| xAI Grok | `grok`, `grok-responses`, `grok-anthropic` |
| OpenRouter | `openrouter` |
| GitHub Copilot | `github-copilot` |

Some services have multiple profiles for different protocols, endpoints, or account types. Azure, Bedrock, and Vertex have protocol adapters for custom profiles; they are not built-in connection presets in the table. Available models and capabilities depend on the profile, your account permissions, and the service.

Qwen, Kimi, and MiniMax use separate profiles, regional API URLs, and credentials for China and international services. Choose the region where the account and key were created; keys are not interchangeable across regions. Beijing, Singapore, US, and Hong Kong Qwen Search profiles expose Responses API knowledge-base search, while `minimax` and `minimax-intl` use Anthropic Messages with MiniMax-hosted Web Search.

See the [File Attachments Guide](docs/file-attachments.md) for sharing images across remote devices, resolver setup, and provider file lifetimes.

Select a usage region with `with_region(Region::ChinaMainland)` or `with_region(Region::International)` before building a client. Provider/model lists, resolution and failover honor this region while keeping the complete configuration. Custom profiles declare `regions`; missing declarations allow both regions. See [region filtering](docs/api.en.md#region-filtering).

Default builds include no tokenizer backends or assets. See [local input token estimates](#6-estimate-input-tokens-offline) for opt-in features. Changes to public extension APIs, usage reports, and configuration v2 are covered in the [architecture and API migration guide](docs/architecture-migration.en.md).

## Getting Started

You need Rust 1.94.0 or later and an API key with access to the target model. This walkthrough sends a first request using the built-in `openai` connection.

Three concepts help orient you: a **provider** is a model service, a **profile** describes a connection (protocol, endpoint, models, and related settings), and a **client** sends requests through profiles. Your application supplies credentials and conversation messages; the client handles protocol conversion, network requests, and response parsing.

### 1. Create an Application and Add Dependencies

```sh
cargo new llm-example
cd llm-example
```

Add these dependencies under `[dependencies]` in `Cargo.toml`:

```toml
[dependencies]
lingxi-llm-client = { git = "https://github.com/lingxi-coder/llm-client", branch = "main" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
serde_json = "1"
```

### 2. Create a Client and Send a Request

Save this complete example as `src/main.rs`:

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, Secret};
use lingxi_llm_client::{builtin_providers, LlmClientBuilder, RequestOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profiles = builtin_providers()?;
    let client = LlmClientBuilder::new(&profiles)?
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()?;
    let options = RequestOptions {
        credential: Some(Secret::new(std::env::var("OPENAI_API_KEY")?)),
        ..RequestOptions::default()
    };
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "gpt-4.1-mini",
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello! Introduce yourself briefly."}]
        }]
    }))?;

    let response = client.chat().complete_in("openai", &request, &options).await?;
    println!("{}", response.message.text());
    Ok(())
}
```

Set the environment variable and run the application to print the model's response:

```sh
export OPENAI_API_KEY="your-api-key"
cargo run
```

The example reads the environment variable explicitly; the client does not load credentials automatically. `openai` is a built-in profile name, and `gpt-4.1-mini` is a model ID in the repository's catalog; your account must have access to it. Use `client.providers()` and `client.chat().models()` to inspect configured connections and models.

Use `client.chat()` for conversations: `complete_in()` selects a connection, `complete()` routes by model, and `stream_in()` / `stream()` return streaming events. `client.chat().models()` filters out models whose metadata declares image output; `client.images().models()` lists the separate image catalog. Existing top-level completion and streaming methods remain available and use the same execution path. See [ChatService](docs/api.en.md#chatservice) and [Streaming Responses](docs/api.en.md#streaming-responses).

#### Stream a Chat Response

To stream the same request, replace the `complete_in()` call and its `println!` in `main` with `stream_chat(&client, &request, &options).await?`. Add this function, merging duplicate imports:

```rust,no_run
use std::io::{self, Write};
use lingxi_llm_client::protocol::{CompletionRequest, StreamEvent};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn stream_chat(
    client: &LlmClient,
    request: &CompletionRequest,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = client.chat().stream_in("openai", request, options).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
                io::stdout().flush()?;
            }
            StreamEvent::End { stop_reason, .. } => {
                eprintln!("\nStop: {stop_reason:?}");
            }
            _ => {}
        }
    }
    eprintln!("Profile: {}", stream.executed_profile());
    eprintln!("Usage: {:?}", stream.usage_report());
    eprintln!("Inference: {:?}", stream.inference_report());
    Ok(())
}
```

`stream_in()` selects the starting profile; `client.chat().stream(request, options)` routes by model. `ModelStream` provides its own asynchronous `next()`, so no `StreamExt` import is needed. Errors can occur both when opening the stream and while reading events; `?` propagates either. Read until `None`, including after an `End` event, then inspect the final reports.

This text-only example ignores tool and reasoning events. Applications that use tools or replay assistant messages must also preserve tool calls, native content, and signatures; see [Streaming Responses](docs/api.en.md#streaming-responses). Usage can be missing or partial; check `usage_is_complete()` before treating it as complete. Once a stream is returned, read failures do not trigger automatic failover. Dropping the stream releases its underlying response. Set `RequestOptions.total_timeout` to bound the entire request; streams have no total deadline by default.

### 3. Configure, Update, and Read Models

To persist configuration, declare the example's `client` as `mut` and call `client.set_config_dir("./config")?` after building it. This loads and persists `providers.json` in that directory; call it again when the application starts to restore saved configuration.

| Task | API |
| --- | --- |
| Add or replace a connection | `add_provider(profile)`; supply a profile name, protocol, endpoint, and models |
| Read connection configuration | `provider("openai")` / `profiles()` |
| List connections and visible models | `providers()` / `chat().models()` |
| Refresh the model catalog from the service | `sync_provider("openai", options.credential.as_ref()).await?` |
| Set a provider's model allowlist | `set_tracked_models(provider_id, model_ids)` |
| Control visibility of a tracked model | `set_model_visibility(profile_name, model_id, visible)` |
| Delete a connection or restore a built-in profile | `remove_provider(profile_name)` / `restore_builtin(profile_name)` |

Sync each account's connection separately with that account's credential. See [Local Persistence and Multiple Accounts](docs/api.en.md#local-persistence-and-multiple-accounts) and [Model Directory](docs/api.en.md#model-directory) for complete examples and update rules.

Configuration uses v2; v1 files return `UnsupportedVersion` and are not migrated automatically. `add_provider()` replaces a complete profile. To override individual fields while inheriting catalog defaults, use `configured_models()` to obtain row IDs and `set_model_override()` / `clear_model_override()`. See the [configuration v2 contract](docs/architecture-migration.en.md).

### 4. Use Web Search

Reuse the `client`, `request`, and `options` from the example, change the message to a question that needs web information, and call `search(&client, &request, &options).await?` inside `main`. Add this function to `src/main.rs`, merging duplicate imports:

```rust,no_run
use lingxi_llm_client::protocol::{CompletionRequest, WebSearchConfig};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn search(
    client: &LlmClient,
    request: &CompletionRequest,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client
        .web_search_in("openai", request, WebSearchConfig::default(), options)
        .await?;
    println!("{}", response.message.text());
    if let Some(search) = response.web_search {
        for citation in search.citations {
            println!("{}", citation.url);
        }
    }
    Ok(())
}
```

The `web_search*` convenience methods belong to `LlmClient`. To use the Chat service directly, set `request.web_search = Some(WebSearchConfig::default())` and call `client.chat().complete_in()` or `client.chat().stream_in()` with the request.

Search is performed by the provider and requires a model that supports it; enabling search does not guarantee every request triggers a search. See the [Web Search Guide](docs/web-search.en.md) for domain restrictions, citations, and streaming search.

### 5. Generate Images

Reuse the client and credential above by calling `generate_image(&client, &options).await?` inside `main`. Add this function, merging duplicate imports. Image requests use their own request types and model catalog:

```rust,no_run
use lingxi_llm_client::protocol::{ImageGenerationRequest, ImageRequestOptions};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn generate_image(
    client: &LlmClient,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = ImageGenerationRequest {
        model: "gpt-image-1.5".into(),
        prompt: "An orange cat on a blue background".into(),
        references: vec![],
        output: Default::default(),
        provider_options: Default::default(),
    };
    let image_options = ImageRequestOptions {
        credential: options.credential.clone(),
        ..Default::default()
    };
    let response = client.images().generate_in("openai", &request, &image_options).await?;
    println!("{} images", response.images.len());
    Ok(())
}
```

Use `client.images().models()` to list image models visible in the selected region. OpenAI, Gemini, Qwen, xAI, MiniMax, GLM/Z.AI, and OpenRouter have built-in image routes; editing, reference images, masks, and native tasks depend on the model.

`submit_in()` returns an `ImageTaskRef`, and `get_task()` queries it once; the application owns polling and task persistence. Save temporary result URLs promptly. Image requests do not automatically retry or fail over, and image usage is separate from Chat token pricing. See the [Image Generation and Editing Guide](docs/images.en.md).

### 6. Estimate Input Tokens Offline

Replace the client dependency above to enable the OpenAI tokenizer:

```toml
[dependencies]
lingxi-llm-client = { git = "https://github.com/lingxi-coder/llm-client", branch = "main", features = ["tokenizer-openai"] }
```

```rust,no_run
use lingxi_llm_client::{LlmClient, LocalTokenCountError, protocol::CompletionRequest};

fn estimate_input(client: &LlmClient, request: &CompletionRequest) -> Result<(), LocalTokenCountError> {
    let estimate = client.estimate_local_tokens_in("openai", request)?;
    println!("{} input tokens via {}", estimate.input_tokens, estimate.tokenizer);
    if estimate.is_partial {
        println!("Not counted: {:?}", estimate.uncounted_components);
    }
    Ok(())
}
```

Other optional features are `tokenizer-deepseek`, `tokenizer-qwen`, `tokenizer-kimi`, and `tokenizer-glm`; `tokenizers-all` enables all five. Only mapped models are supported: a disabled backend returns `FeatureDisabled`, and an unsupported model returns `UnsupportedModel`.

Estimates make no network requests and do not read attachments. Images, remote files, and hidden provider state can make an estimate partial; it does not replace provider-reported usage. See [Local Input Token Estimates](docs/api.en.md#local-input-token-estimates) for coverage and limitations.

### 7. Troubleshoot Errors

Classify errors with `LlmError::kind()` or the error variants. Do not branch on `message` text.

| Error | What to check |
| --- | --- |
| `BuildError` | Check for duplicate `profile_name`, a missing codec for the protocol, an unregistered authenticator, or an invalid pricing window. |
| `ModelUnavailable` / route ambiguity | Verify the model ID exists in the profile. Use `resolve()` to inspect routing or select a connection with `complete_in()` / `stream_in()`. |
| `Authentication` / `PermissionDenied` | Confirm `RequestOptions.credential` belongs to the starting connection. Put failover credentials in `fallback_credentials`, keyed by backup profile name, and check account permissions. |
| `InvalidRequest` / `UnsupportedCapability` | Check request fields, provider configuration, and whether the selected model supports tools, search, or other requested capabilities. |
| `RateLimited` / `QuotaExceeded` | Check `retry_after` and account quota. Failover is off by default; the client does not retry the same connection automatically. |
| `Transport` / `TransportTimeout` / `TlsCert` | Check the endpoint, network connectivity, timeout, and TLS certificate chain. |
| `StreamInterrupted` | The stream ended unexpectedly. Check network and provider status; do not blindly replay a request after partial output. |
| `ProviderStoreError` | Check configuration directory permissions, `providers.json` syntax, and model directory request or disk-write errors. |

See [Error Handling](docs/api.en.md#error-handling) for error details. An undeclared Web Search adapter, protocol mismatch, or unsupported option also returns an error before sending the request; see the [search support matrix](docs/web-search.en.md#support-matrix-and-parameters).

## Documents

| Document | Contents |
| --- | --- |
| [API Guide](docs/api.en.md) | Client construction, requests, streams, authentication, configuration, model catalogs, and costs |
| [Reasoning and Pricing](docs/inference.en.md) | Capability discovery, thinking budgets, effort, service tiers, and actual cost estimates |
| [Image Generation and Editing](docs/images.en.md) | Image models, generation, editing, reference images, and native tasks |
| [Architecture and API Migration](docs/architecture-migration.en.md) | Extension contracts, usage reports, configuration v2, and tokenizer features |
| [Local Input Token Estimates](docs/api.en.md#local-input-token-estimates) | Offline counting, supported models, and uncounted components |
| [Account Balances and Token Usage](docs/api.en.md#account-balances-and-token-usage) | Account identity, query budgets, quota windows, and partial reports |
| [File Attachments Guide](docs/file-attachments.md) | Stable attachment references, remote display, resolver setup, and provider file input |
| [Web Search Guide](docs/web-search.en.md) | Support matrix, search options, citations, stream events, and context replay |
| [Extension APIs](docs/api.en.md#extension-apis) | Adding protocols, model directories, and custom authentication |
| [Catalog Maintenance and Publishing](docs/maintenance.en.md) | Updating built-in provider and model catalogs and publishing the crate |
| [Review Notes and Known Boundaries](docs/review.en.md) | Resolved issues, design decisions, and responsibilities of the calling application |

Run `cargo doc --no-deps --open` in this repository to browse Rust type and method documentation.

## Development

Build the project from source when making changes:

```sh
git clone https://github.com/lingxi-coder/llm-client.git
cd llm-client
cargo build --locked
```

Built-in provider profiles live in `data/providers/*.toml`. Add a profile for an existing protocol; implement and register `WireCodec` for a new protocol, and `ModelDirectory` if catalog sync is needed. See [Extension APIs](docs/api.en.md#extension-apis).

Before submitting changes, run:

```sh
cargo test --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked
cargo test --manifest-path tests/downstream-docs/Cargo.toml --locked --offline --target-dir target
```

CI also tests the `tokenizers-all` configuration and compiles each tokenizer feature separately. When changing token estimation, additionally run `cargo test --locked --features tokenizers-all`.

Tests use mocked transport and local loopback HTTP servers; no real API key is required.

## License

This project is licensed under either the MIT License or Apache License 2.0, at your option. See [MIT License](LICENSE-MIT) and [Apache License 2.0](LICENSE-APACHE).
