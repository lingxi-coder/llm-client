# Anthropic Programmatic Tool Calling

[简体中文](anthropic-programmatic-tools.md)

Programmatic Tool Calling lets code inside Anthropic's Code Execution container invoke function tools declared on the request. This client encodes the request, exposes caller metadata, and retains native state for replay. Tool permissions, execution, and results remain the host's responsibility.

## Request

Set `allowed_callers` on each function in `ChatRequest.tools` and add `AnthropicHostedTool::CodeExecution` to the same request:

```rust
use lingxi_llm_client::providers::anthropic::types::{AnthropicCodeExecutionConfig, AnthropicToolCaller};
use lingxi_llm_client::protocol::{ChatRequest, ToolSpec};
use serde_json::json;

let mut request: ChatRequest = serde_json::from_value(json!({
    "model": "claude-opus-5-5",
    "messages": [{"role":"user","content":[{"type":"text","text":"Look up recent records."}]}]
})).unwrap();
request.hosted_tools.push(lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
    AnthropicCodeExecutionConfig::default(),
).into());
request.tools.push(ToolSpec {
    tool_type: None,
    extra: serde_json::Value::Null,
    name: "lookup".into(),
    description: "Look up records".into(),
    input_schema: json!({
        "type": "object",
        "properties": { "query": { "type": "string" } },
        "required": ["query"]
    }),
    strict: false,
    defer_loading: false,
    native_options: Vec::new(),
}.with_anthropic_allowed_callers(vec![AnthropicToolCaller::CodeExecution20260120]));
```

`allowed_callers` accepts `Direct`, `CodeExecution20260120`, and `CodeExecution20260521`. Anthropic treats the two Code Execution versions as interchangeable. An omitted or empty list keeps the usual direct caller behavior; including `Direct` with a Code Execution caller allows either path. This field is supported on the first-party Anthropic Messages route and on a typed Microsoft Foundry deployment hosted on Anthropic, when the selected underlying model supports Programmatic Tool Calling. Azure-hosted Foundry deployments are rejected. On Foundry, `model` remains the custom deployment name; the model allowlist uses `FoundryDeployment.model_id`.

The client preflights the documented constraints: Code Execution must be present on the same request; the selected model must support Programmatic Tool Calling; strict tools and recursive local `$ref` schemas are rejected for programmatic callers; a `tool_choice` that forces a programmatic-only tool is rejected; and a profile-level `disable_parallel_tool_use: true` override cannot be combined with Programmatic Tool Calling. Anthropic documents support for Fable 5/5.1, Mythos 5/5.1, Opus 4.5/4.6/4.7/4.8/5/5.5, and Sonnet 4.5/4.6/5. Haiku 4.5 and Mythos Preview accept Code Execution but are excluded from Programmatic Tool Calling. Foundry support additionally requires the model ID to be documented for the selected hosting option.

`allowed_callers` guides model behavior; it is not an authorization boundary. The host must enforce its own tool permissions and must not run untrusted commands merely because the provider used Code Execution.

## Responses and continuation

For non-streaming responses, Anthropic's `caller` object is retained on `ContentBlock::ToolUse.caller`. For streams, it is available on `StreamEvent::ToolCallDelta.caller`. The raw JSON preserves `type`, `tool_id`, and future fields through serialization and replay. A programmatic caller's `tool_id` points to the preceding native `server_tool_use` block. That server block remains `ProviderContent`; it is not duplicated as a client-executed tool call.

To resume a pending programmatic call, replay the assistant message, return the matching `ToolResult`, resend the original `ToolSpec` definitions, and provide the response's scoped Code Execution container reference. The client checks that caller metadata refers to an earlier server-tool block and that the tool definition allows a Code Execution caller. It does not execute the function or code, choose permissions, or store conversation history.

Malformed known caller shapes fail before dispatch while remaining intact in the decoded response. Direct and unknown caller objects are replayed unchanged through Anthropic Messages wire families, including the first-party API, Bedrock, Vertex, Azure AI Foundry, and compatible gateway profiles such as OpenRouter. Recognized Programmatic Tool Calling callers are accepted on the first-party API or on a supported Anthropic-hosted Foundry deployment with the matching typed model identity; this does not enable the feature on Azure-hosted Foundry, Bedrock, Vertex, or gateways. Other protocol codecs, such as OpenAI and Gemini, reject caller metadata instead of dropping it.

Container continuation must use a reference scoped to the same Foundry resource, account identity, deployment, and underlying model. Foundry Code Execution files use the separate Foundry Files service and are scoped to the resource and account, not to a container, deployment, or model. Custom Skills can be uploaded and managed through the separate Skills service; their Foundry references are also scoped to the resource and account. Foundry does not support Skill version-content downloads. Do not send a first-party Anthropic file reference to a Foundry resource.

See [Code Execution](anthropic-code-execution.en.md) for container scope, uploaded files, output artifacts, and failure behavior. The implementation is covered by mock wire tests only; real account availability and hosted execution have not been tested.

References: Anthropic's [Programmatic Tool Calling guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling), [Tool Reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference), [Code Execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool), and [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry).
