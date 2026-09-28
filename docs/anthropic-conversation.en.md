# Anthropic mid-conversation instructions and tools

The first-party Anthropic Messages and Vertex Claude codecs support `MessageRole::System` in conversation history on the documented Fable 5/5.1, Mythos 5/5.1, Opus 4.8, Opus 5, and Opus 5.5 wire model IDs. Unsupported models and compatible gateways are rejected locally. Vertex enables the documented message controls and tool references; Foundry and Bedrock mid-conversation controls remain separate implementation tasks.

Consecutive system messages form one section. The section follows a user message (including tool results), or an assistant message ending in a server tool result, and precedes an assistant message or the end of the array. Text can follow a paused server-tool turn; tool changes cannot. Keep initial instructions in `ChatRequest.system`.

```rust
use lingxi_llm_client::protocol::{ChatRequest, ContentBlock, ConversationMessage, MessageRole};
# fn append(request: &mut ChatRequest) {
request.messages.push(ConversationMessage {
    native_options: Vec::new(),
    role: MessageRole::System,
    content: vec![ContentBlock::Text {
        text: "Use explicit units in subsequent answers.".into(),
        thought_signature: None,
    }],
});
# }
```

Tool additions and removals use Anthropic `ProviderContent` blocks in a system message. The codec selects `inline-tools-2026-09-15`; plain text does not need a beta. Custom tool definitions and references retain their wire structure. An inline definition can update an existing tool of the same type. A removal takes a reference, not a definition.

```rust
use lingxi_llm_client::protocol::{ContentBlock, ConversationMessage, MessageRole, ProtocolFamily};
use serde_json::json;
let message = ConversationMessage {
    native_options: Vec::new(),
    role: MessageRole::System,
    content: vec![ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: json!({
            "type": "tool_addition",
            "tool": {
                "type": "tool_definition",
                "definition": {
                    "name": "lookup",
                    "description": "Look up an identifier",
                    "input_schema": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"]
                    }
                }
            }
        }),
    }],
};
```

A definition's `cache_control` is counted at its message position. Put it on the addition block or the definition, never both. Deferred definitions cannot carry a breakpoint. Existing four-breakpoint and TTL-order checks apply. The client does not rewrite previous messages or manage cache/history lifetimes.

Turn-scoped `clear_at` and per-message effort are supported as described below. Anthropic-defined server and client tools must use their typed feature configuration; a generic `tool_definition` cannot bypass their model, route, or retry checks. Typed Code Execution and Web Fetch can opt into by-value placement with `inline_definition: true` on the corresponding config. The request must contain an exact matching `tool_addition` definition; the codec keeps the typed config for model and route checks, pins requests against automatic retry, and omits only that native tool from the top-level `tools` array. This placement currently requires the first-party Claude API. Tool Search remains in top-level `tools` with the full deferred catalog. The Claude API does support surfacing a known top-level deferred tool later using a `tool_addition` by reference. The Tool Search guide says searchable definitions remain in top-level `tools`; it does not say that a by-value inline definition joins that catalog, so this client does not claim that behavior. MCP toolsets also require a matching `AnthropicMcpConfig`. Tool execution, approval, history retention and credential refresh remain the host's responsibility. Validation here is offline and mock-based, not a live account acceptance test.

Source: [Anthropic mid-conversation system messages and tool changes](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages).

## Inline MCP servers

Keep the server in `hosted_tools`, mark its toolset as inline, and append the generated system message at the moment its tools become available. The server URL remains in `mcp_servers`; the toolset is not duplicated in the top-level `tools` array. Authorization still uses `RequestOptions.mcp_authorizations` keyed by server name. The codec sends both required betas.

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicMcpConfig};
use lingxi_llm_client::protocol::{ChatRequest, };
# fn append(request: &mut ChatRequest) -> Result<(), Box<dyn std::error::Error>> {
let server = AnthropicMcpConfig::new("calendar", "https://mcp.example.com/calendar")?
    .with_inline_toolset(true);
request.messages.push(server.inline_tool_addition_message()?);
request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(server).into());
# Ok(())
# }
```

`AnthropicToolChange` and `AnthropicToolReference` also build custom definitions and reference additions/removals as native blocks or system messages. Inline custom definitions reuse the Messages strict-schema subset, reference checks, and request-wide complexity limits; an exact repeated name/schema pair counts once, while a changed schema still counts in the same request. `allowed_callers` accepts `direct`, `code_execution_20260120`, or `code_execution_20260521`. Programmatic calls require typed Code Execution configuration on the same request and a supported model. Recursive `$ref` and `strict: true` are unsupported with programmatic calling. Historical calls are checked against the tool definition active at that position, and a pending call requires the latest history to retain a definition that still allows Code Execution; removing it or changing it to direct-only fails before dispatch. Typed Code Execution and Web Fetch support by-value placement when the exact typed definition appears at the correct point in the conversation and the corresponding config enables `inline_definition`. Other Anthropic server/client tools cannot use a generic inline definition, preserving feature-specific validation. The provider remains authoritative for rendered tool text size and remotely discovered MCP tool counts, which cannot be known locally.

Sources: [strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use), [programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling), and [tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference).

## Turn-scoped reminders and message effort

Use `ConversationMessage::with_anthropic_options()` for message-level controls. This metadata is translated into wire `clear_at` and `output_config.effort`, never a content block. Other built-in protocol routes reject these controls. Rust message literals now include `native_options: Vec::new()` unless they use these options; prefer the message constructors for ordinary messages.

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicClearAt, AnthropicMessageOptions};
use lingxi_llm_client::protocol::{ConversationMessage};
let reminder = ConversationMessage::system_text("Request independent reads together.")
    .with_anthropic_options(AnthropicMessageOptions {
        clear_at: Some(AnthropicClearAt::NextUserMessage),
        effort: None,
    });
```

`NextUserMessage` requires one or more text blocks, normal system-message placement, and no effort, tool changes, or cache breakpoint. A later user turn, including tool results, ends the reminder's rendering on the server. Keep the original message unchanged in history; the client never removes it. Explicit `Never` preserves lasting-message semantics. Both explicit values select the clear-at beta. Automatic caching is forwarded to Anthropic, which chooses its own eligible breakpoint.

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicMessageEffort, AnthropicMessageOptions};
use lingxi_llm_client::protocol::{ConversationMessage};
let mut change = ConversationMessage::system_text("")
    .with_anthropic_options(AnthropicMessageOptions {
        clear_at: None,
        effort: Some(AnthropicMessageEffort::Low),
    });
change.content.clear();
```

Effort-only messages with empty content may appear anywhere; a consecutive system group that includes text or tool changes follows the ordinary placement rules. Per-message effort is limited to Fable 5.1, Mythos 5.1, Opus 5.5 and Opus 5. Supported levels are low, medium, high, xhigh and max. The output-config beta is automatic. This changes effort at the next user message, including a tool-result message; `InferenceReport.requested_effort` follows that boundary without overwriting the top-level request setting. Effort and `NextUserMessage` cannot be combined.

See [per-message effort](https://platform.claude.com/docs/en/build-with-claude/effort#per-message-effort-beta). The same controls are supported on this library's Vertex Claude route.

## Vertex tool references

Vertex tool additions/removals reference tools already declared at the top level and use `mid-conversation-tool-changes-2026-07-01`. The first-party inline-definition beta is not automatically sent to Vertex; inline definitions and inline MCP are rejected. The independent message-control betas are unchanged. See the [Vertex guide](anthropic-vertex.en.md).
