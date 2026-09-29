# llm-client

[简体中文](README.md)

`llm-client` is a standalone Rust library for calling different LLM services from an application. Applications use shared request and response types and choose models and connections through provider profiles; the client handles protocol encoding, HTTP transport, stream parsing, and error classification. It also supports model catalogs, provider-hosted Web Search, usage tracking, and cost estimation.

Protocol types are defined by this crate and available through `lingxi_llm_client::protocol`. Applications own tool execution, permissions, conversation history, credential storage and refresh scheduling, and context compaction; see the [API guide](docs/api.en.md#host-tool-execution-and-context-recovery).

Version **0.3.0** organizes implementations under `providers/<provider_id>`. Use `client.chat()`, `client.images()`, and `client.embeddings()` for common operations; bind `client.provider::<OpenAiClient>("openai")?` for resources such as `provider.audio()`, `provider.batches()`, and `provider.retrieval()`. Provider-specific types live with their provider. This release removes old entry points and module paths without compatibility aliases. See [the 0.3.0 architecture](docs/architecture-migration.en.md) and [pinned official SDK references](docs/provider-sdk-references.md).

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
- **Provider authentication protocols**: `auth::oauth` provides explicit Anthropic, OpenAI, and Copilot login/refresh requests. Applications own browser interaction, credential storage, and scheduling.
- **Built-in defaults, extensible interfaces**: HTTP transport and authenticators are included; replace transport or clocks, or add protocols. Applications supply credentials per request; secrets are not persisted in provider configuration.

### Supported LLM Providers

The repository includes these connection presets in [`data/providers/`](data/providers/). Each name in the table is a profile name accepted by methods such as `complete_in()`.

| Service | Built-in profiles |
| --- | --- |
| <img src="docs/assets/providers/openai.png" width="20" height="20" alt="OpenAI icon"> [OpenAI](https://developers.openai.com/api/docs) | `openai` |
| <img src="docs/assets/providers/anthropic.png" width="20" height="20" alt="Anthropic icon"> [Anthropic](https://platform.claude.com/docs) | `anthropic` |
| <img src="docs/assets/providers/gemini.png" width="20" height="20" alt="Google Gemini icon"> [Google Gemini](https://ai.google.dev/gemini-api/docs) | `gemini` |
| <img src="docs/assets/providers/deepseek.png" width="20" height="20" alt="DeepSeek icon"> [DeepSeek](https://api-docs.deepseek.com) | `deepseek`, `deepseek-search` |
| <img src="docs/assets/providers/kimi.png" width="20" height="20" alt="Kimi icon"> [Kimi](https://platform.kimi.com/docs) | `kimi`, `kimi-intl`, `kimi-code`, `kimi-search`, `kimi-search-intl` |
| <img src="docs/assets/providers/qwen.png" width="20" height="20" alt="Qwen / Model Studio icon"> [Qwen / Model Studio](https://help.aliyun.com/en/model-studio/) | `qwen`, `qwen-intl`, `qwen-us`, `qwen-hk`, `qwen-search`, `qwen-search-intl`, `qwen-search-us`, `qwen-search-hk` |
| <img src="docs/assets/providers/minimax.png" width="20" height="20" alt="MiniMax icon"> [MiniMax](https://platform.minimax.io/docs) | `minimax`, `minimax-intl` |
| <img src="docs/assets/providers/zai.svg" width="20" height="20" alt="Z.AI / GLM icon"> [Z.AI / GLM](https://docs.z.ai/) | `zai`, `zai-coding`, `glm`, `glm-coding` |
| <img src="docs/assets/providers/xai.svg" width="20" height="20" alt="xAI Grok icon"> [xAI Grok](https://docs.x.ai/) | `grok`, `grok-responses`, `grok-anthropic` |
| <img src="docs/assets/providers/openrouter.png" width="20" height="20" alt="OpenRouter icon"> [OpenRouter](https://openrouter.ai/docs) | `openrouter` |
| <img src="docs/assets/providers/github-copilot.svg" width="20" height="20" alt="GitHub Copilot icon"> [GitHub Copilot](https://docs.github.com/en/copilot) | `github-copilot` |

Some services have multiple profiles for different protocols, endpoints, or account types. Azure, Bedrock, and Vertex have protocol adapters for custom profiles; they are not built-in connection presets in the table. Available models and capabilities depend on the profile, your account permissions, and the service.

Qwen, Kimi, and MiniMax use separate profiles, regional API URLs, and credentials for China and international services. Choose the region where the account and key were created; keys are not interchangeable across regions. Beijing, Singapore, US, and Hong Kong Qwen Search profiles expose Responses API knowledge-base search, while `minimax` and `minimax-intl` use Anthropic Messages with MiniMax-hosted Web Search.

See the [File Attachments Guide](docs/file-attachments.md) for sharing images across remote devices, resolver setup, and provider file lifetimes.

Select a usage region with `with_region(Region::ChinaMainland)` or `with_region(Region::International)` before building a client. Provider/model lists, resolution and failover honor this region while keeping the complete configuration. Custom profiles declare `regions`; missing declarations allow both regions. See [region filtering](docs/api.en.md#region-filtering).

Default builds include no tokenizer backends or assets. See [local input token estimates](#6-estimate-input-tokens-offline) for opt-in features. Changes to public extension APIs, usage reports, and configuration v3 are covered in the [0.3.0 architecture and API guide](docs/architecture-migration.en.md).

### Capability support

The tables describe **implemented client adapters and services** in this working tree, not every upstream feature or every model. Provider names mean that an integration exists; availability still depends on the profile, model, region and account. Unlisted providers are not claimed as supported. Native services use their own APIs and routes; Chat compatibility alone does not enable them. Real-account validation is still pending. Each guide below links to the relevant official API documentation.

#### Chat and tools

| Capability | Purpose | Providers / scope | Documentation |
| --- | --- | --- | --- |
| Chat, streaming and function tools | Normalize text, tool calls and usage events; the application executes tools | All built-in providers, subject to the selected model | [API](docs/api.en.md) |
| Reasoning and service tiers | Control thinking budgets, effort and fast; inspect the executed tier | Capability queries cover all built-in providers; budget, effort and fast are declared separately per model | [Reasoning](docs/inference.en.md) |
| JSON and schema output | Constrain output and validate JSON/schema or deserialize into Rust types | OpenAI, Anthropic, Gemini; OpenRouter by upstream model; verify JSON Object and Schema separately for other compatible services | [Output contracts](docs/services.en.md) |
| Web search and citations | Search the web on the provider and return sources and citations | OpenAI, Anthropic, Gemini, OpenRouter, GLM/Z.AI, MiniMax, Kimi Search, Qwen Search, DeepSeek Search; xAI requires a custom Responses search profile | [Search matrix](docs/web-search.en.md) |
| Explicit prompt caching | Reuse tool, system or message prefixes to reduce repeated input costs | OpenAI Responses (native GPT-5.6+ options and independent documented retention); Anthropic / Messages; MiniMax supports five-minute breakpoints only. Server-side automatic caching is separate | [OpenAI Responses](docs/openai-responses-prompt-cache.en.md) · [Prompt cache overview](docs/services.en.md) |
| Remote context cache | Create, read, update and delete reusable context cache resources | Gemini (independent cachedContents service) | [Gemini Cache](docs/gemini-context-cache.en.md) |
| Gateway response caching | Reuse complete responses and read explicit server-reported HIT/MISS | OpenRouter Chat, Responses, Messages, and Embeddings | [OpenRouter Cache](docs/services.en.md) |
| Stateful continuation | Continue an existing response with an account-bound reference | OpenAI Responses; Gemini Interactions uses a separate API | [Responses](docs/services.en.md) · [Interactions](docs/interactions.en.md) |
| Hosted code execution and containers | Run model-generated code in remote containers and manage container files | OpenAI Responses; first-party Anthropic and explicitly Anthropic-hosted Foundry execution/container reuse | [Code Interpreter](docs/services.en.md) · [Containers](docs/openai-containers.en.md) · [Anthropic](docs/anthropic-code-execution.en.md) |
| Programmatic tool calling | Receive function calls initiated by remote code execution for the host to execute and return | Supported Anthropic models with code execution; caller metadata preserved | [Programmatic tools](docs/anthropic-programmatic-tools.en.md) |
| Anthropic Skills requests | Load built-in or already-uploaded Skills in remote code execution containers | First-party and Anthropic-hosted Foundry; custom references bound to a workspace or resource account; independent upload and version management | [Skills and containers](docs/anthropic-code-execution.en.md) · [Skills resources](docs/anthropic-skills.en.md) |
| Remote MCP and tool search | Connect remote MCP servers or discover deferred function tools | OpenAI Responses: MCP; Tool Search on GPT-5.4+; Anthropic: native MCP and Tool Search; Gemini Interactions: tool requests subject to model/agent restrictions | [OpenAI Tool Search](docs/openai-tool-search.en.md) · [OpenAI MCP](docs/openai-hosted-extended.en.md) · [Anthropic Tool Search](docs/anthropic-tools.en.md) · [Anthropic MCP](docs/anthropic-mcp.en.md) · [Anthropic Web Fetch](docs/anthropic-web-fetch.en.md) · [Browser / Computer](docs/anthropic-client-toolsets.en.md) · [Vertex Claude](docs/anthropic-vertex.en.md) · [Foundry Claude](docs/anthropic-foundry.en.md) · [Mid-conversation instructions and tools](docs/anthropic-conversation.en.md) · [Gemini](docs/interactions.en.md) |

#### Retrieval, files and jobs

| Capability | Purpose | Providers / scope | Documentation |
| --- | --- | --- | --- |
| Files and multimodal attachments | Manage uploads and file lifetimes; resolve application attachments into model inputs | OpenAI, Anthropic, Gemini, Qwen, MiniMax, xAI and others, by model/media/purpose; file management does not imply Chat file-reference support | [Attachments](docs/file-attachments.md) |
| Text embeddings | Turn text into vectors for semantic search, clustering and similarity | Built-in routes for OpenAI, Gemini, OpenRouter and GLM; Qwen requires an explicit workspace endpoint | [Embeddings](docs/services.en.md) · [Model parameter limits](docs/embedding-limits.en.md) |
| Multimodal embeddings | Embed a combination of text and media | Gemini Embedding 2 | [Gemini Embedding](docs/gemini-embedding.en.md) |
| Knowledge bases and file retrieval | Manage remote indexes/documents and retrieve relevant content for RAG | OpenAI Vector Stores, Gemini File Search, GLM Knowledge Base, Qwen Beijing workspaces, xAI Collections; operation coverage varies | [OpenAI](docs/retrieval.en.md) · [Gemini](docs/gemini-file-search.en.md) · [GLM](docs/glm-knowledge.en.md) · [Qwen](docs/qwen-knowledge.en.md) · [xAI](docs/xai-collections.en.md) |
| Retrieval reranking | Reorder candidate documents by relevance to a query | Qwen Beijing workspaces, OpenRouter | [Qwen](docs/qwen-rerank.en.md) · [OpenRouter](docs/openrouter-rerank.en.md) |
| Batch jobs | Submit many requests offline, inspect status and read individual results | OpenAI, Anthropic, Gemini, Qwen, Kimi, OpenRouter, xAI, mainland GLM; models, regions and cancellation differ | [OpenAI](docs/batches.en.md) · [Anthropic](docs/anthropic-batch.en.md) · [Gemini](docs/gemini-batch.en.md) · [Qwen](docs/qwen-batch.en.md) · [Kimi](docs/kimi-batch.en.md) · [OpenRouter](docs/openrouter-batch.en.md) · [xAI](docs/xai-batch.en.md) · [GLM](docs/glm-batch.en.md) |
| Background and asynchronous inference | Submit long-running inference, fetch results and resume streams where supported | OpenAI Background, Gemini Interactions, xAI Deferred, GLM Async; xAI results are consumed once | [OpenAI](docs/background.en.md) · [Gemini](docs/interactions.en.md) · [xAI](docs/deferred.en.md) · [GLM](docs/glm-async.en.md) |

#### Images and speech

| Capability | Purpose | Providers / scope | Documentation |
| --- | --- | --- | --- |
| Image generation | Generate images from prompts or supported reference images | OpenAI, Gemini, Qwen, xAI, MiniMax, GLM/Z.AI, OpenRouter; Wan requires a custom workspace route | [Images](docs/images.en.md) |
| Image editing and native tasks | Edit source images, apply masks, or submit and query native generation tasks | Editing: OpenAI, Gemini, Qwen, xAI, OpenRouter; masks: OpenAI; tasks: Qwen, GLM/Z.AI, custom Wan | [Images](docs/images.en.md) |
| Speech recognition (ASR / STT) | Transcribe recordings, with timestamps or speaker information where available | OpenAI, MiniMax, OpenRouter, xAI, GLM/Z.AI cloud; Qwen asynchronous file and standalone realtime transcription; separate self-hosted GLM-ASR adapter | [OpenAI](docs/audio.en.md) · [MiniMax](docs/minimax-audio.en.md) · [OpenRouter](docs/openrouter-audio.en.md) · [xAI](docs/xai-audio.en.md) · [GLM Cloud](docs/glm-cloud-audio.en.md) · [Qwen](docs/qwen-asr.en.md) · [GLM Self-hosted](docs/glm-audio.en.md) |
| Audio translation | Translate speech in an audio file into English text | OpenAI Whisper | [Audio](docs/audio.en.md) |
| Speech synthesis (TTS) | Convert text to audio bytes, streams or temporary URLs | OpenAI, Gemini, Vertex Gemini, MiniMax, OpenRouter, xAI, Qwen, mainland GLM; MiniMax also has asynchronous long-text, ordinary and bidirectional WebSocket TTS | [OpenAI](docs/audio.en.md) · [Gemini](docs/gemini-speech.en.md) · [Vertex](docs/vertex-speech.en.md) · [MiniMax](docs/minimax-tts.en.md) · [Async TTS](docs/minimax-async-tts.en.md) · [OpenRouter](docs/openrouter-audio.en.md) · [xAI](docs/xai-audio.en.md) · [Qwen](docs/qwen-tts.en.md) · [GLM](docs/glm-cloud-audio.en.md) |
| Chat audio | Send audio in Chat and receive native audio deltas | OpenRouter, validated against model input/output modalities | [Chat Audio](docs/openrouter-chat-audio.en.md) |
| Realtime bidirectional sessions | Exchange low-latency audio, text and tool events; the application owns audio devices | OpenAI Realtime, Gemini Live, xAI Voice, mainland GLM; built-in WebSocket transport requires realtime-websocket | [OpenAI](docs/realtime.en.md) · [Gemini](docs/gemini-live.en.md) · [xAI](docs/xai-realtime.en.md) · [GLM](docs/glm-realtime.en.md) |

#### Configuration and usage

| Capability | Purpose | Providers / scope | Documentation |
| --- | --- | --- | --- |
| Model catalogs, regions and account routing | Manage visible models, separate accounts, region filtering and optional failover | All built-in providers; remote catalog sync depends on the profile's directory API | [API](docs/api.en.md) · [Shared client](docs/client-reuse.en.md) |
| Usage and cost reports | Read token/cache usage and the executed connection; estimate costs from known prices | Mapped usage across built-in providers; missing usage/prices are not treated as zero | [Reasoning and pricing](docs/inference.en.md) |
| Account limits and balances | Query balances, historical usage, quota windows or plan entitlements | DeepSeek, Kimi, OpenRouter, Qwen, MiniMax; OpenAI, Anthropic, xAI admin APIs; Codex, Copilot and Kimi Code require host RPC integration. Credentials and permissions vary | [Account queries](docs/api.en.md#account-balances-and-token-usage) |
| Offline token estimates | Estimate visible text input without networking and report uncounted content | Opt-in OpenAI, DeepSeek, Qwen, Kimi and GLM tokenizer backends; mapped models only | [Local estimates](docs/api.en.md) |

For model-level official evidence and unknown/unsupported states, see [OpenAI / Gemini](docs/capability-matrix-openai-gemini.en.md), [China providers](docs/capability-matrix-china.en.md), and [Anthropic / xAI / OpenRouter / Copilot](docs/capability-matrix-west.en.md). Evidence does not imply implementation or live validation; remaining work is tracked in the [implementation status](docs/implementation-plan.md).

Provider icons above come from official sites, with local copies and attribution in [icon sources](docs/assets/providers/README.md).

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
use lingxi_llm_client::protocol::{ChatRequest, Secret};
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
    let request: ChatRequest = serde_json::from_value(serde_json::json!({
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

Use `client.chat()` for conversations: `complete_in()` selects a connection, `complete()` routes by model, and `stream_in()` / `stream()` return streaming events. `client.chat().models()` filters out models whose metadata declares image output; `client.images().models()` lists the separate image catalog. Old top-level completion and streaming methods have been removed. See [ChatService](docs/api.en.md#chatservice) and [Streaming Responses](docs/api.en.md#streaming-responses).

#### Stream a Chat Response

To stream the same request, replace the `complete_in()` call and its `println!` in `main` with `stream_chat(&client, &request, &options).await?`. Add this function, merging duplicate imports:

```rust,no_run
use std::io::{self, Write};
use lingxi_llm_client::protocol::{ChatRequest, StreamEvent};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn stream_chat(
    client: &LlmClient,
    request: &ChatRequest,
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

Build a long-lived client once and share `client.clone()` between concurrent tasks; model, effort, Fast tier, and credentials stay request-scoped. For persistent or dynamic configuration, replace `.build()?` above with `.build_managed()?`, bind the result as `let (client, config) = ...`, and call `config.set_config_dir("./config").await?`. This loads and persists `providers.json`; repeat it at application startup to restore saved configuration. Updates publish a new snapshot for subsequent requests while in-flight operations retain their original configuration. See [client reuse and migration](docs/client-reuse.en.md).

| Task | API |
| --- | --- |
| Add or replace a connection | `config.add_provider(profile).await?`; supply a profile name, protocol, endpoint, and models |
| Read connection configuration | `snapshot.profile("openai")` / `snapshot.profiles()` |
| List connections and visible models | `client.providers()` / `client.chat().models()` |
| Refresh the model catalog from the service | `config.sync_provider("openai", options.credential.as_ref()).await?` |
| Set a provider's model allowlist | `config.set_tracked_models(provider_id, model_ids).await?` |
| Control visibility of a tracked model | `config.set_model_visibility(profile_name, model_id, visible).await?` |
| Delete a connection or restore a built-in profile | `config.remove_provider(profile_name).await?` / `config.restore_builtin(profile_name).await?` |

Capture `let snapshot = client.snapshot();` before borrowing profiles. Use that same snapshot for preflight, requests, and pricing when all steps must use one configuration version. Sync each account's connection separately with that account's credential. See [Local Persistence and Multiple Accounts](docs/api.en.md#local-persistence-and-multiple-accounts) and [Model Directory](docs/api.en.md#model-directory) for complete examples and update rules.

Configuration uses v3; v1/v2 files return `UnsupportedVersion` and are not migrated automatically. `add_provider()` replaces a complete profile. To override individual fields while inheriting catalog defaults, use `configured_models()` to obtain row IDs and `set_model_override()` / `clear_model_override()`. See the [configuration v3 contract](docs/architecture-migration.en.md).

### 4. Use Web Search

Reuse the `client`, `request`, and `options` from the example, change the message to a question that needs web information, and call `search(&client, &request, &options).await?` inside `main`. Add this function to `src/main.rs`, merging duplicate imports:

```rust,no_run
use lingxi_llm_client::protocol::{ChatRequest, WebSearchConfig};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn search(
    client: &LlmClient,
    request: &ChatRequest,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client
        .chat().web_search_in("openai", request, WebSearchConfig::default(), options)
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

The `web_search*` convenience methods belong to `ChatService`. To use the Chat service directly, set `request.set_hosted_web_search(Some(WebSearchConfig::default()))` and call `client.chat().complete_in()` or `client.chat().stream_in()` with the request.

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
use lingxi_llm_client::{LlmClient, LocalTokenCountError, protocol::ChatRequest};

fn estimate_input(client: &LlmClient, request: &ChatRequest) -> Result<(), LocalTokenCountError> {
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
| [Architecture and API Migration](docs/architecture-migration.en.md) | Extension contracts, usage reports, configuration v3, and tokenizer features |
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

## Service documentation

- Chat and tools: [Output contracts](docs/services.en.md) · [Gemini Interactions](docs/interactions.en.md) · [Anthropic tool search](docs/anthropic-tools.en.md) · [Anthropic Code Execution](docs/anthropic-code-execution.en.md) · [OpenAI Tool Search](docs/openai-tool-search.en.md) · [OpenAI hosted tools and MCP](docs/openai-hosted-extended.en.md) · [xAI Remote MCP](docs/xai-remote-mcp.en.md) · [Qwen hosted Code Interpreter](docs/qwen-hosted.en.md) · [Qwen Web Extractor](docs/qwen-web-extractor.en.md) · [OpenRouter server tools](docs/openrouter-server-tools.en.md) · [OpenRouter Chat audio](docs/openrouter-chat-audio.en.md) · [Realtime](docs/realtime.en.md) · [OpenAI GPT-Live](docs/openai-live.en.md) · [Gemini Live](docs/gemini-live.en.md) · [xAI Realtime](docs/xai-realtime.en.md) · [GLM Realtime](docs/glm-realtime.en.md) · [Qwen Realtime](docs/qwen-realtime.en.md) · [Qwen LiveTranslate](docs/qwen-translate.en.md)
- Retrieval and vectors: [Managed retrieval](docs/retrieval.en.md) · [Gemini multimodal embeddings](docs/gemini-embedding.en.md) · [Gemini File Search](docs/gemini-file-search.en.md) · [GLM Knowledge Base](docs/glm-knowledge.en.md) · [Qwen knowledge retrieval](docs/qwen-knowledge.en.md) · [Qwen Rerank](docs/qwen-rerank.en.md) · [OpenRouter Rerank](docs/openrouter-rerank.en.md) · [xAI Collections](docs/xai-collections.en.md)
- Batch and asynchronous work: [OpenAI Batch](docs/batches.en.md) · [Anthropic Batch](docs/anthropic-batch.en.md) · [Gemini Batch](docs/gemini-batch.en.md) · [Qwen Batch](docs/qwen-batch.en.md) · [OpenRouter Batch](docs/openrouter-batch.en.md) · [Kimi Batch](docs/kimi-batch.en.md) · [xAI Batch](docs/xai-batch.en.md) · [GLM Batch](docs/glm-batch.en.md) · [OpenAI Background](docs/background.en.md) · [xAI Deferred Chat](docs/deferred.en.md) · [GLM asynchronous inference](docs/glm-async.en.md) · [Gemini explicit cache](docs/gemini-context-cache.en.md) · [Qwen Prompt Cache](docs/qwen-prompt-cache.en.md) · [OpenRouter Prompt Cache](docs/openrouter-prompt-cache.en.md) · [OpenAI Containers](docs/openai-containers.en.md)
- Speech: [OpenAI Audio](docs/audio.en.md) · [Gemini Speech](docs/gemini-speech.en.md) · [Vertex TTS](docs/vertex-speech.en.md) · [Gemini Chat audio](docs/gemini-chat-audio.en.md) · [Gemini Voices](docs/gemini-voices.en.md) · [MiniMax ASR](docs/minimax-audio.en.md) · [MiniMax TTS](docs/minimax-tts.en.md) · [MiniMax voices](docs/minimax-voices.en.md) · [MiniMax WebSocket TTS](docs/minimax-streaming-tts.en.md) · [MiniMax Bidi TTS](docs/minimax-bidi-tts.en.md) · [MiniMax asynchronous TTS](docs/minimax-async-tts.en.md) · [OpenRouter Audio](docs/openrouter-audio.en.md) · [xAI Audio](docs/xai-audio.en.md) · [xAI streaming TTS](docs/xai-streaming-tts.en.md) · [xAI custom voices](docs/xai-custom-voices.en.md) · [xAI realtime STT](docs/xai-stt.en.md) · [Qwen asynchronous transcription](docs/qwen-asr.en.md) · [Qwen Realtime ASR](docs/qwen-asr-realtime.en.md) · [Qwen TTS](docs/qwen-tts.en.md) · [Qwen Audio Generation](docs/qwen-audio-generation.en.md) · [Qwen Realtime TTS](docs/qwen-tts-realtime.en.md) · [Self-hosted GLM-ASR](docs/glm-audio.en.md) · [GLM hosted audio](docs/glm-cloud-audio.en.md)
- [Implementation status](docs/implementation-plan.md) · [OpenAI/Gemini capability evidence matrix](docs/capability-matrix-openai-gemini.en.md) · [China provider capability evidence matrix](docs/capability-matrix-china.en.md) · [Other provider capability evidence matrix](docs/capability-matrix-west.en.md)

[Live provider acceptance and read-only checks](docs/provider-acceptance.en.md)
