# Anthropic Code Execution and container reuse

[简体中文](anthropic-code-execution.md)

`AnthropicHostedTool::CodeExecution` enables execution in Anthropic's hosted container through the first-party Messages API or a typed Microsoft Foundry deployment hosted on Anthropic. The adapter sends the fixed tool definition `{"type":"code_execution_20260521","name":"code_execution"}`. Code Execution itself does not require a beta header. See Anthropic's [Code Execution guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool) and [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry).

## Create and reuse a container

`AnthropicCodeExecutionConfig::default()` leaves `container` unset, allowing the provider to create one. A returned `ChatResponse::anthropic_container()` contains the complete native container object in its public `envelope` field. Use `reference_for(scope)` to explicitly bind its ID to the original connection and account before reuse; `AnthropicContainerRef::new(id, scope)` is also available when importing an ID saved by the application.

This example assumes the client routes `claude-opus-5-5` to its configured first-party Anthropic profile. Supply the already-valid credential for that account. The application must handle a paused, truncated, or client-tool response before starting a new user turn. The Foundry route and its separately scoped container reference are described below.

```rust,no_run
use lingxi_llm_client::providers::anthropic::types::{AnthropicCodeExecutionConfig, AnthropicContainerScope};
use lingxi_llm_client::{
    protocol::{ChatRequest, ChatResponse,
        ConversationMessage, Secret, StopReason},
    LlmClient, RequestOptions,
};

# async fn example(
#     client: &LlmClient,
#     credential: Secret<String>,
# ) -> Result<ChatResponse, Box<dyn std::error::Error>> {
const MODEL: &str = "claude-opus-5-5";
const ACCOUNT: &str = "anthropic-workspace-main";
let options = RequestOptions {
    credential: Some(credential),
    account_scope: Some(ACCOUNT.into()),
    ..Default::default()
};
let mut request: ChatRequest = serde_json::from_value(serde_json::json!({
    "model": MODEL,
    "max_tokens": 4096,
    "messages": []
}))?;
request.messages.push(ConversationMessage::user_text(
    "Use code execution to write the number 37 to /tmp/number.txt.",
));
request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
    AnthropicCodeExecutionConfig::default(),
).into());

let response = client.chat().complete(&request, &options).await?;
if response.stop_reason != StopReason::EndTurn {
    // Return this response to the application's turn-handling loop.
    return Ok(response);
}
let profile = response.executed_profile.as_deref()
    .ok_or("response has no executed profile")?;
let scope = AnthropicContainerScope::new(
    profile, "https://api.anthropic.com", ACCOUNT, MODEL,
)?;
let reference = response.anthropic_container()
    .ok_or("provider returned no execution container")?
    .reference_for(scope)?;

// Preserve the original assistant content, including native execution blocks.
request.messages.push(response.message);
request.messages.push(ConversationMessage::user_text(
    "Read /tmp/number.txt and use code execution to calculate its square.",
));
request.hosted_tools = vec![lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
    AnthropicCodeExecutionConfig {
        container: Some(reference),
        ..Default::default()
    },
).into()];
let follow_up = client.chat().complete(&request, &options).await?;
println!("{}", follow_up.message.text());
# Ok(follow_up)
# }
```

The reusable reference checks the profile name, normalized endpoint, stable non-secret account identity, and exact resolved request model before dispatch. Use the canonical wire model ID in the scope; a catalog alias is resolved before validation. Without Skills, only the container ID is sent as the top-level `container`; with Skills, the wire value is an object containing `id` and `skills`. All local scope fields stay local. The application is responsible for supplying a truthful account identity for its credential. `file_account_scope` is a separate option and does not satisfy this check.

Reuse requires `RequestOptions.account_scope`. Callers encoding requests directly must instead supply the same account through `CodecContext::with_account_scope(Some(account))`. A request for a new container can omit account scope. Neither a response ID nor `ChatRequest.continuation` substitutes for an Anthropic container reference or for the message history the caller needs to retain.

## Microsoft Foundry

Anthropic documents Code Execution and Programmatic Tool Calling on Microsoft Foundry only for deployments **Hosted on Anthropic**. The Foundry resource base URL is `https://{resource}.services.ai.azure.com/anthropic`; Messages requests go to `/v1/messages`, and the request's `model` remains the deployment name. Add `FoundryDeployment` to the selected model row so the client can validate hosting and the underlying model ID. The `claude-mythos-preview` model is supported for Code Execution on the Claude API and Foundry, but this does not enable Programmatic Tool Calling for that preview. Azure-hosted deployments are rejected before dispatch.

For container reuse, bind the response ID to the Foundry resource, selected deployment, underlying model, and the same caller-supplied account identity. The container ID is the only local scope value sent to Messages:

```rust,no_run
use lingxi_llm_client::providers::anthropic::types::{AnthropicContainerRef, AnthropicContainerScope};
use lingxi_llm_client::protocol::{ChatResponse, FoundryDeployment,
    FoundryHosting};

# fn bind(response: &ChatResponse) -> Result<AnthropicContainerRef, Box<dyn std::error::Error>> {
let foundry_scope = AnthropicContainerScope::new_foundry(
    "foundry-claude",
    "https://example-resource.services.ai.azure.com/anthropic",
    "foundry-resource-account",
    "my-opus-deployment",
    FoundryDeployment {
        hosting: FoundryHosting::Anthropic,
        model_id: "claude-opus-5-5".into(),
    },
)?;
let foundry_container = response
    .anthropic_container()
    .ok_or("provider returned no execution container")?
    .reference_for(foundry_scope)?;
# Ok(foundry_container)
# }
```

The constructor rejects an Azure-hosted identity and stores the Foundry resource endpoint without converting it to `api.anthropic.com`. A reference cannot be reused after changing its profile, endpoint, `account_scope`, deployment name, hosting, or underlying model.

## Use Agent Skills

Agent Skills are configured in the same top-level `container` field as Code Execution. Anthropic's API accepts up to 20 Skills per request; each reference has a source (`anthropic` or `custom`), a `skill_id`, and an optional version. The Skills API is GA, so the client does not add the old `skills-2025-10-02` beta header. Skills still require Code Execution and a model in its documented compatibility list. Built-in Skills and custom Skills work with supported Anthropic-hosted Foundry deployments; custom Skills are uploaded and managed through the separate Skills service and scoped to the Foundry resource and account. Foundry does not support downloading Skill version content.

```rust,no_run
use lingxi_llm_client::providers::anthropic::types::{AnthropicCodeExecutionConfig, AnthropicSkillRef, AnthropicSkillScope};
use lingxi_llm_client::protocol::{ChatRequest, };

# fn add_skills(request: &mut ChatRequest) -> Result<(), Box<dyn std::error::Error>> {
let workspace = AnthropicSkillScope::new(
    "anthropic",
    "https://api.anthropic.com",
    "anthropic-workspace-main",
)?;
let mut execution = AnthropicCodeExecutionConfig::default();
execution.skills = vec![
    AnthropicSkillRef::anthropic("pptx").with_version("latest"),
    AnthropicSkillRef::custom("skill_01AbCdEfGhIjKlMnOpQrStUv", workspace)
        .with_version("skver_01AbCdEfGhIjKlMnOpQrStUv"),
];
request
    .hosted_tools
    .push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(execution).into());
# Ok(())
# }
```

Anthropic-managed IDs include `pptx`, `xlsx`, `docx`, and `pdf`; pin them with an eight-digit catalog version or use `latest`. Built-in Skills are also available to supported Anthropic-hosted Foundry deployments. Custom IDs must come from a Skill uploaded through the independent [Skills resource service](anthropic-skills.en.md). Use `AnthropicSkillScope::new` for the first-party API or `AnthropicSkillScope::new_foundry` for a Foundry resource explicitly hosted on Anthropic; custom refs are checked against the matching profile and account scope, and Foundry refs are resource scoped rather than deployment scoped. Use a distinct stable `account_scope` for each resource or workspace so a custom ID cannot accidentally be replayed under another identity. Custom versions accept current `skver_…` IDs, legacy `skill_version_…` IDs, or `latest`; omitting `version` is also supported. Foundry supports Skill CRUD and version metadata, but not version-content downloads.

When continuing with a reused container and Skills, include the desired Skills again so the next request has the same Skill set. The scoped Skill identity remains local and is never serialized into the provider request. See Anthropic's [Agent Skills API guide](https://platform.claude.com/docs/en/build-with-claude/skills-guide), [Skills quickstart](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart), and [Code Execution model compatibility](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#compatibility).

## Supply uploaded files

Upload inputs through the existing `FileService`, then put each `ProviderFileRef::model_reference()` in `AnthropicCodeExecutionConfig.files`. When building an initial execution request, use this configuration in place of the default configuration above:

```rust,no_run
use lingxi_llm_client::providers::anthropic::types::{AnthropicCodeExecutionConfig};
use lingxi_llm_client::{
    files::ProviderFileRef,
    protocol::{HostedTool},
    RequestOptions,
};

# fn file_input(uploaded: &ProviderFileRef) -> (HostedTool, RequestOptions) {
let execution: lingxi_llm_client::protocol::HostedTool = lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(AnthropicCodeExecutionConfig {
    files: vec![uploaded.model_reference()],
    ..Default::default()
}).into();
let options = RequestOptions {
    // Use the same non-secret identity supplied to FileService for this upload.
    file_account_scope: Some("anthropic-workspace-main".into()),
    ..Default::default()
};
# (execution, options)
# }
```

The encoder appends native `container_upload` blocks to the last user message. A nonempty `files` list therefore requires the last message to have the user role. Every dispatch checks the file's provider, profile, endpoint fingerprint, protocol, and account binding; `RequestOptions.file_account_scope`, or `CodecContext::with_file_scope` for direct encoding, must match. Duplicate IDs in `files`, and typed file IDs already present in native `container_upload` blocks, are rejected before sending. Container reuse still needs the separate `account_scope` described above. The caller supplies the request credential as usual.

`ProviderFileRef::model_reference()` preserves the original file `expires_at` in `ProviderFileSource`. The shared file preflight rejects malformed expiry or a known expiry at or before the current time, and the client rechecks before sending; the timestamp is not sent in the model-input wire. Missing expiry is allowed and does not guarantee availability. The reference also preserves raw processing status. Only documented Gemini/Qwen states are currently interpreted; their status rules do not establish Anthropic file availability. A known-expiry failure does not automatically refresh or reupload the file. See [File attachment validation](file-attachments.md) for timestamp formats and validation timing. Anthropic's [file storage expiration](https://platform.claude.com/docs/en/build-with-claude/files#file-expiration) is distinct from the execution container's rolling `expires_at`, which continues to cause no local container-expiry rejection.

The chat request does not upload or download files automatically. Use `FileService` for upload and generated-file retrieval, and retain its scoped references. Native `container_upload` blocks already present in history continue to replay unchanged. See [File attachments and lifecycle](file-attachments.md).

For Foundry, construct the service with `FileService::new_foundry_for_model(http, profile, selected_model, authenticator, credential, account_scope)`. It requires a model row from that profile with an explicit `FoundryDeployment` hosted on Anthropic and the supported Foundry resource URL. File operations use the resource route under `/anthropic/v1/*`, including `/anthropic/v1/files`. The returned reference records `FoundryClaude`, the canonical resource endpoint, and the same non-secret `account_scope`; pass its `model_reference()` into `AnthropicCodeExecutionConfig.files` as above.

Foundry file references are resource/account scoped, not model or deployment scoped, so another deployment on the same resource and account can use the same file if its selected model supports Code Execution. Container references keep the stricter deployment/model scope described earlier. Azure-hosted deployments, first-party `api.anthropic.com` file references, and references from another resource or account are rejected before dispatch. A known-expired `expires_at` is rejected locally, but the client does not refresh or reupload the file and cannot guarantee it remains available through inference. The Foundry guide documents the Files API only for Anthropic hosting, and the Files API reference documents the `/v1/files` operations.

## Streaming and native content

`ModelStream::anthropic_container()` exposes the current observed metadata from `message_start.message.container` and `message_delta.delta.container`. Poll the stream to receive updates; the accessor does not consume frames or wait for completion. The native object, including unknown fields, remains available in `envelope`. The [Messages reference](https://platform.claude.com/docs/en/api/typescript/messages) documents the response and stream container fields.

```rust,no_run
use lingxi_llm_client::{protocol::ChatRequest, LlmClient, RequestOptions};

# async fn inspect_stream(
#     client: &LlmClient,
#     request: &ChatRequest,
#     options: &RequestOptions,
# ) -> Result<(), Box<dyn std::error::Error>> {
let mut stream = client.chat().stream(request, options).await?;
while let Some(event) = stream.next().await {
    let event = event?;
    // Deliver each event to the application's transcript/stream consumer.
    println!("{event:?}");
    if let Some(container) = stream.anthropic_container() {
        // The host may retain this observation with the request's actual scope.
        println!("{}", container.envelope);
    }
}
# Ok(())
# }
```

Metadata observed before an interrupted stream is not evidence that execution completed. The application decides whether to resume or inspect existing state; the client does not automatically repeat the request.

`ChatResponse::anthropic_usage() -> Option<&serde_json::Value>` preserves the original raw usage object. `ModelStream::anthropic_usage()` exposes raw usage with field updates folded across received frames; the original frames remain in `ProviderEvent`. The reported `usage.server_tool_use.code_execution_requests` maps to the cross-provider `ServerToolUsage.code_interpreter_requests` count. This count does not determine container runtime charges or organization free-tier eligibility, and the client does not infer that execution is free from it.

Server execution calls and results remain `ContentBlock::ProviderContent`, preserving IDs, inputs, outputs, and generated-file metadata for native replay. They do not appear in `ConversationMessage::tool_uses()` and must not be executed by the host. Ordinary client tools continue to use `ToolUse` and the application's execution flow. See [Preserving Anthropic-native content](anthropic-native-content.en.md) for raw stream events, fragmented inputs, and replay behavior.

`pause_turn` remains `StopReason::Other("pause_turn")`. The caller decides whether to continue by appending the returned assistant message unchanged and reusing the container reference. This is a new, explicit continuation request; the client does not loop automatically or translate the pause into `EndTurn`.

## Routing and failure behavior

- The first-party profile must declare `provider_id = "anthropic"`, use `ProtocolFamily::AnthropicMessages`, and normalize to `https://api.anthropic.com`. Foundry instead requires `ProtocolFamily::FoundryClaude`, a Foundry resource URL, and an explicit `FoundryDeployment` on the selected model row. A compatible protocol or arbitrary proxy URL alone does not enable either route.
- The exact Code Execution model allowlist includes Fable 5/5.1, Mythos 5/5.1, Opus 4.5/4.6/4.7/4.8/5/5.5, Sonnet 4.5/4.6/5, Haiku 4.5, and Mythos Preview where the platform documents it. Programmatic Tool Calling excludes Haiku 4.5 and Mythos Preview. Foundry model support is the intersection with its documented deployment model IDs and requires Anthropic hosting. Unknown identities fail before sending; account and region availability remain provider-managed.
- An execution request is not automatically retried or sent to a failover profile, whether it creates or reuses a container. A transport failure may occur after the server has already executed commands, so an application retry is an explicit decision.
- Expired-container errors are returned to the caller. The client neither silently creates a replacement nor rejects reuse based on `expires_at`: Anthropic documents this as a rolling timestamp, distinct from the container's 30-day lifetime. See [Container reuse](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#container-reuse).

See [Anthropic Programmatic Tool Calling](anthropic-programmatic-tools.en.md) for `allowed_callers`, caller metadata, and continuation constraints. This slice supports prebuilt and custom Skills on the first-party API and Anthropic-hosted Foundry; use the independent [Skills resource service](anthropic-skills.en.md) for CRUD, version upload, and listing. Foundry Skill references are scoped to the resource and account; Foundry version-content downloads remain unsupported. Native tool error blocks remain inspectable in response content. Mock and wire validation do not establish live account availability or successful execution on Anthropic's service.
