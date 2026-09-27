# Anthropic Claude on Microsoft Foundry

[简体中文](anthropic-foundry.md)

Foundry routes the Anthropic Messages wire through your Foundry resource endpoint. Use the explicit `foundry_claude` protocol and base URL **https://<resource>.services.ai.azure.com/anthropic**. Ordinary Messages requests need no extra Foundry model identity. Tool Search, Web Fetch and Code Execution do: set `ModelProfile.foundry` on the exact model row because the request model is a deployment name and cannot identify its underlying model or hosting choice.



A model row keeps four identities separate:

| Field | Purpose |
| --- | --- |
| `request_model` | The Foundry deployment name sent in the Messages body as the model |
| `foundry.hosting` | The hosting selected when the Foundry deployment was created: **azure** or **anthropic** |
| `foundry.model_id` | The exact underlying Claude model ID used for feature validation |
| `billing_model` | The pricing lookup key; it is not used to infer the model |

Do not derive hosting or model identity from a deployment name, display name, billing ID, endpoint, or Azure resource. If multiple profile rows share a deployment name but provide different identities, select the intended row with `CodecContext::for_model`; an ambiguous wire-only context is rejected for identity-sensitive tools. Identity metadata is not sent in the request body.

## Tool Search

Anthropic hosted Tool Search works on both Foundry hosting options when the exact underlying model is documented for both Foundry and Tool Search. The codec checks the intersection of those model tables. It does not infer compatibility from a family prefix.

| Foundry hosting | Accepted underlying model IDs |
| --- | --- |
| Azure | `claude-opus-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-haiku-4-5` |
| Anthropic | `claude-fable-5-1`, `claude-mythos-5-1`, `claude-fable-5`, `claude-mythos-5`, `claude-opus-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5` |

For 4.5 Foundry deployments, use the documented undated Foundry IDs such as `claude-opus-4-5`, `claude-sonnet-4-5`, and `claude-haiku-4-5`. Tool Search is not listed for Sonnet 5, even though Foundry offers that model. Availability can vary by subscription and deployment; provider access remains authoritative.

The request model stays the deployment name. This example performs local request validation and encoding only; it makes no network call and configures no credential:

~~~rust
use lingxi_llm_client::protocol::{
    AnthropicToolSearchConfig, AnthropicToolSearchStrategy, ChatRequest, FoundryDeployment,
    FoundryHosting, HostedTool, ProviderProfile,
};
use lingxi_llm_client::{
    CodecContext, EncodeRequest, FoundryClaudeCodec, RequestMode, WireCodec,
};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "my-foundry-resource",
        "profile_name": "foundry-claude",
        "protocol": "foundry_claude",
        "base_url": "https://my-resource.services.ai.azure.com/anthropic",
        "auth": "none",
        "models": [{
            "display_model": "Claude Opus",
            "request_model": "prod-opus-55",
            "billing_model": "accounting-opus"
        }]
    }))?;
    profile.models[0].foundry = Some(FoundryDeployment {
        hosting: FoundryHosting::Anthropic,
        model_id: "claude-opus-5-5".into(),
    });

    let mut request: ChatRequest = serde_json::from_value(json!({
        "model": "prod-opus-55",
        "messages": [{"role":"user","content":[{"type":"text","text":"Find a calendar tool"}]}]
    }))?;
    request.hosted_tools.push(HostedTool::AnthropicToolSearch(
        AnthropicToolSearchConfig {
            strategy: AnthropicToolSearchStrategy::Bm25,
        },
    ));

    let context = CodecContext::new(&profile, "prod-opus-55", RequestMode::Complete);
    FoundryClaudeCodec.validate_request(&request, &context)?;
    let wire = FoundryClaudeCodec.encode_request(EncodeRequest::new(&request), &context)?;
    let body: Value = serde_json::from_slice(&wire.body)?;
    assert_eq!(body["model"], "prod-opus-55");
    assert_eq!(body["tools"][0]["type"], "tool_search_tool_bm25_20251119");
    Ok(())
}
~~~

## Web Fetch

The typed Web Fetch adapter supports the Foundry endpoint and requires an explicit per-row Foundry identity. Azure-hosted deployments accept only `web_fetch_20250910` (basic fetch). They accept direct callers only; non-direct Code Execution callers are rejected because Azure hosting does not expose Code Execution / Programmatic Tool Calling (PTC). Anthropic-hosted deployments accept all four versions described in the [Web Fetch guide](anthropic-web-fetch.en.md); versions with dynamic filtering additionally require an eligible underlying model. Choosing direct-only callers disables dynamic filtering when that version supports it. The codec rejects unsupported host/version/model combinations before sending.

## MCP connector

Foundry supports the typed remote MCP connector on both hosting options using `mcp-client-2025-11-20`. The base connector does not require ModelProfile.foundry or a model allowlist. The current public Foundry documentation establishes the base `mcp-client-2025-11-20` connector, but does not establish availability of the newer `mcp-client-2026-09-15` beta features used for pinned lists, mcp_tool_listing, or inline MCP changes. This client therefore rejects those newer operations on Foundry, including Anthropic-hosted deployments; leave the pinned list unset and do not replay or add those native blocks. See the [MCP connector guide](anthropic-mcp.en.md) for a typed configuration example and the remaining request contract.

~~~rust
use lingxi_llm_client::protocol::{
    AnthropicMcpConfig, ChatRequest, FoundryDeployment, FoundryHosting, HostedTool, ProviderProfile,
};
use lingxi_llm_client::{
    CodecContext, EncodeRequest, FoundryClaudeCodec, RequestMode, WireCodec,
};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "my-foundry-resource",
        "profile_name": "foundry-claude",
        "protocol": "foundry_claude",
        "base_url": "https://my-resource.services.ai.azure.com/anthropic",
        "auth": "none",
        "models": [{
            "display_model": "Claude Opus",
            "request_model": "prod-opus-55",
            "billing_model": "accounting-opus"
        }]
    }))?;
    // The identity field is shown for completeness; base MCP does not require it.
    profile.models[0].foundry = Some(FoundryDeployment {
        hosting: FoundryHosting::Azure,
        model_id: "claude-opus-5-5".into(),
    });

    let mut request: ChatRequest = serde_json::from_value(json!({
        "model": "prod-opus-55",
        "messages": [{"role":"user","content":[{"type":"text","text":"Search the docs server"}]}]
    }))?;
    request.hosted_tools.push(HostedTool::AnthropicMcp(
        AnthropicMcpConfig::new("docs", "https://mcp.example.com/sse")?,
    ));

    let context = CodecContext::new(&profile, "prod-opus-55", RequestMode::Complete);
    FoundryClaudeCodec.validate_request(&request, &context)?;
    let wire = FoundryClaudeCodec.encode_request(EncodeRequest::new(&request), &context)?;
    let body: Value = serde_json::from_slice(&wire.body)?;
    assert_eq!(body["model"], "prod-opus-55");
    let beta = wire.headers.iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| value.as_str())
        .unwrap_or("");
    assert!(beta.split(',').any(|value| value == "mcp-client-2025-11-20"));
    Ok(())
}
~~~

Foundry’s model catalog, feature availability, authentication methods, and supported deployment regions can change. The application supplies credentials and owns Microsoft Entra token acquisition and refresh; this codec only encodes the request. The examples here are pure local encoding checks, not tests against an Azure resource or Anthropic account.

References: [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry), [Microsoft Foundry deployment and authentication guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-claude), [Anthropic Tool Search compatibility](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool), [Anthropic Web Fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool), and [Anthropic MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector).


## Code Execution and programmatic tools

`HostedTool::AnthropicCodeExecution` uses the selected row's explicit hosting and underlying model identity. Only supported models hosted on Anthropic accept execution; Azure-hosted deployments fail preflight. The wire `model` remains the deployment name. Programmatic tool callers additionally follow their own documented model restrictions.

Complete and streaming responses retain native container and usage metadata. Import a returned container ID with `AnthropicContainerScope::new_foundry`, binding the profile, exact resource endpoint, stable account identity, deployment name and `FoundryDeployment`; changing any of those prevents reuse. An uncertain execution outcome does not trigger automatic retry or failover. See [Code Execution](anthropic-code-execution.en.md) for the supported file and Skill boundaries.
