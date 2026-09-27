# OpenRouter server tools

This client exposes OpenRouter's regex tool search and hosted shell as typed
`HostedTool` variants. OpenRouter executes these tools inside the request; they
are not `ToolUse` calls for the host to run. The provider can call them zero or
more times before returning its answer. The feature is available through the
OpenRouter Responses and Messages APIs. OpenRouter documents tool search for
any model and provider. Shell also works with any model, but only on the global
`openrouter.ai` endpoint; OpenRouter rejects it on the regional endpoints and
on Chat Completions. See the [server-tools overview](https://openrouter.ai/docs/guides/features/server-tools),
[tool-search contract](https://openrouter.ai/docs/guides/features/server-tools/tool-search),
and [shell contract](https://openrouter.ai/docs/guides/features/server-tools/shell).

Configure a profile with `provider_id = "openrouter"`, protocol
`OpenAiResponses` or `AnthropicMessages`, and base URL
`https://openrouter.ai/api/v1`. The model must resolve through that profile's
directory. Pass its credential through `RequestOptions` when required; these
examples receive caller-prepared options and do not look up credentials.

## Tool search

Add `HostedTool::OpenRouterToolSearch` and set `ToolSpec.defer_loading` on the
function tools to hide until the model searches for them. A deferred function
remains a caller-executed tool after discovery: its eventual `ToolUse` still
belongs to the host. `max_results` defaults to OpenRouter's default and is
capped at 50. This client represents the documented regex search variant; it
does not expose OpenRouter's separate Anthropic aliases or the unsupported
BM25 variant.

Deferred tools require OpenRouter tool search or the existing provider-native
Anthropic deferral path. For OpenRouter-managed deferral, use automatic tool
choice. The API also accepts an `allowed_tools` choice, but its schema is not
specified in the public contract; this client does not model it and rejects a
non-auto choice when deferred tools use OpenRouter search. Requests that use
Anthropic-native search keep their existing Anthropic tool-choice behavior.

```rust,no_run
use lingxi_llm_client::protocol::{
    ChatRequest, ConversationMessage, HostedTool, OpenRouterToolSearchConfig, ToolChoice, ToolSpec,
};
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::LlmError;
use serde_json::{json, Value};

pub async fn discover_a_client_tool(
    client: &LlmClient,
    options: &RequestOptions,
) -> Result<(), LlmError> {
    let request = ChatRequest {
        prompt_cache: Default::default(),
        output_format: Default::default(),
        model: "openai/gpt-5.2".into(),
        anthropic_client_toolsets: Vec::new(),
        hosted_tools: vec![HostedTool::OpenRouterToolSearch(
            OpenRouterToolSearchConfig {
                max_results: Some(10),
            },
        )],
        continuation: None,
        system: vec![],
        messages: vec![ConversationMessage::user_text("Find the weather tool and check Tokyo.")],
        tools: vec![ToolSpec {
            name: "get_weather".into(),
            description: "Get the current weather for a city.".into(),
            input_schema: json!({
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
            }),
            strict: false,
            defer_loading: true,
            allowed_callers: vec![],
        }],
        tool_choice: ToolChoice::Auto,
        max_tokens: None,
        temperature: None,
        thinking: None,
        service_tier: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    };
    let _response = client
        .chat()
        .complete(&request, options)
        .await?;
    Ok(())
}
```

## Hosted shell

`HostedTool::OpenRouterShell` sends `type: "openrouter:shell"`. Omitted
parameters leave OpenRouter's defaults in effect: `engine: "auto"`, an
ephemeral `container_auto` environment, a 120-second per-command timeout,
16,384 output characters per stream, and networking disabled. Explicit
`timeout_ms` cannot exceed 300,000 and `max_output_length` cannot exceed
65,536. The model supplies the command list; OpenRouter allows up to 100
commands per call. The server clamps the documented numeric limits and rejects
calls over the command cap.

The optional network policy is either disabled or an allowlist of at most 50
lowercase hostnames/glob patterns. Entries do not include schemes, paths, or
ports. An omitted policy disables outbound access. A container's policy is
fixed when it starts, so the same policy must be sent when reusing that
container.

On the Messages API, shell calls and results are returned as provider-native
`server_tool_use` and `openrouter_shell_tool_result` content blocks. On
Responses, they are provider-native output items. The codecs retain and replay
these blocks as `ProviderContent`; they never surface server-side shell
commands as host `ToolUse` calls. Stream callers should preserve the
`ProviderContent` blocks and may inspect the accompanying raw `ProviderEvent`
frames. Messages streaming reports a returned container envelope in the raw
`message_start` event; the stream's terminal event does not reconstruct a
`ChatResponse.openrouter_container` field.

### Reusing a container

OpenRouter scopes persistent containers by account and workspace. This client
requires a caller-imported `OpenRouterContainerRef` to carry a local
`OpenRouterContainerScope` for the exact profile name, endpoint, and stable
`RequestOptions.account_scope`. Supply the same `account_scope` on each request
that reuses the reference. The scope is local metadata: only the provider's
container ID is sent over the wire.

The Messages `ChatResponse.openrouter_container` value retains the provider's
raw envelope. It is not automatically trusted or attached to an account. The
caller can explicitly associate that observed ID with the profile, endpoint,
and account it just used:

```rust,no_run
use lingxi_llm_client::protocol::{
    ChatRequest, HostedTool, OpenRouterContainerScope,
};
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::LlmError;

pub async fn run_in_a_reused_container(
    client: &LlmClient,
    response: &lingxi_llm_client::protocol::ChatResponse,
    account_scope: &str,
    options: &RequestOptions,
) -> Result<(), LlmError> {
    let scope = OpenRouterContainerScope::new(
        "openrouter-global", // use the selected profile's exact name
        "https://openrouter.ai/api/v1",
        account_scope,
    )?;
    let Some(container) = response
        .openrouter_container
        .as_ref()
        .map(|metadata| metadata.reference_for(scope))
        .transpose()?
    else {
        return Ok(());
    };
    let request: ChatRequest = serde_json::from_value(serde_json::json!({
        "model":"openai/gpt-5.2",
        "messages":[{"role":"user","content":[{"type":"text","text":"Continue."}]}],
        "hosted_tools":[{
            "type":"openrouter_shell",
            "config":{"environment":{"type":"container_reference","container":container}}
        }]
    }))
    .map_err(|error| LlmError::InvalidRequest {
        message: error.to_string(),
    })?;
    let scoped_options = RequestOptions {
        account_scope: Some(account_scope.to_owned()),
        ..options.clone()
    };
    let _response = client.chat().complete(&request, &scoped_options).await?;
    Ok(())
}
```

`OpenRouterContainerMetadata::reference_for` only imports a string ID; the host
owns the decision that the returned envelope belongs to the scope it supplies.
This client does not create containers out of band, poll them, or retry a
request that may already have executed server tools. A server-tool request is
pinned to one route attempt because resending after an interrupted response
could repeat a provider-side action.

`openrouter:bash` is a separate Messages-only server tool and is not part of
this API. Credentials continue to come from the selected OpenRouter profile
and `RequestOptions`; the server tools do not introduce a separate credential
field.
