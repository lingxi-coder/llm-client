# Vertex Claude tools and mid-conversation controls

`VertexClaudeCodec` uses the explicit `vertex_claude` protocol without assuming a particular custom provider_id label. The model stays in the publisher URL; the body carries `anthropic_version: "vertex-2023-10-16"`. Streaming uses `streamRawPredict`, and ordinary requests use `rawPredict`. Explicit API-version overrides are preserved. The caller configures Google authentication, project and location.

This is a pure encoding example: `auth: none` sends no request. Actual calls require Google credentials through the caller's authentication configuration/Authenticator.

```rust
use lingxi_llm_client::{CodecContext, EncodeRequest, RequestMode, VertexClaudeCodec, WireCodec};
use lingxi_llm_client::protocol::{
    AnthropicToolSearchConfig, AnthropicToolSearchStrategy, ChatRequest, HostedTool, ProviderProfile,
};
use serde_json::json;
let profile: ProviderProfile = serde_json::from_value(json!({
    "provider_id": "google-vertex", "profile_name": "vertex",
    "protocol": "vertex_claude", "auth": "none",
    "base_url": "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/locations/us-central1",
    "models": [{"display_model":"claude-opus-5-5", "request_model":"claude-opus-5-5", "billing_model":"claude-opus-5-5"}]
})).unwrap();
let mut request: ChatRequest = serde_json::from_value(json!({
    "model":"claude-opus-5-5",
    "messages":[{"role":"user","content":[{"type":"text","text":"Find the right tool"}]}]
})).unwrap();
request.hosted_tools.push(HostedTool::AnthropicToolSearch(AnthropicToolSearchConfig {
    strategy: AnthropicToolSearchStrategy::Regex,
}));
let context = CodecContext::new(&profile, "claude-opus-5-5", RequestMode::Complete);
let wire = VertexClaudeCodec.encode_request(EncodeRequest::new(&request), &context).unwrap();
assert!(wire.url.ends_with("/publishers/anthropic/models/claude-opus-5-5:rawPredict"));
```

## Tool Search

Regex/BM25 use native stable tool versions without a new beta. Claude API and Vertex validate models against the current official table. Older Vertex models use the published `@YYYYMMDD` IDs rather than Claude API dated forms or display names. Claude API also accepts the three documented 4.5 short aliases. Bedrock InvokeModel preserves opaque model IDs/ARNs and leaves the underlying model decision to the provider. Regional and account availability still requires live validation.

## Mid-conversation System and tool references

System and clear_at support Fable 5/5.1, Mythos 5/5.1 and Opus 4.8/5/5.5. Per-message effort supports Fable 5.1, Mythos 5.1 and Opus 5/5.5. Use exact wire IDs; arbitrary suffixes are not normalized. Message-group placement, turn-scoped reminders and caching restrictions match the first-party protocol.

The lookup tool below must already be declared in request.tools, and the message must follow a permitted user/server-result position. Vertex reference additions/removals use `mid-conversation-tool-changes-2026-07-01`. The first-party inline-definition beta is not automatically applied to Vertex; inline definitions and MCP additions remain rejected.

```rust
use lingxi_llm_client::protocol::{
    AnthropicToolChange, AnthropicToolReference, ChatRequest, ConversationMessage, MessageRole,
};
# fn withdraw_declared_tool(request: &mut ChatRequest) {
request.messages.push(ConversationMessage {
    role: MessageRole::System,
    content: vec![AnthropicToolChange::remove(AnthropicToolReference::tool("lookup")).into_content_block()],
    anthropic: None,
});
# }
```

clear_at and per-message effort carry their independent beta headers. The client preserves history; effort activates at the next user message and appears in request inference reports. The host still owns reminder removal and session recovery. Both codec and high-level preflight reject invalid combinations before attachment reads.

## Stable client toolsets

The stable 20260801 Browser/Computer toolsets use the same declarations, model limits, cache ordering and toolset_name round trip on Vertex as on the first-party route. See [client toolsets](anthropic-client-toolsets.en.md). The host owns execution, authorization and browser/desktop state.

The current Google Cloud platform and tool guides both list Browser/Computer support, superseding the conflicting early audit note. First-party Web Fetch, Code Execution and MCP retain independent platform gates; sharing an encoder does not enable them. Foundry hosting modes and custom deployment model identity remain separate work.

Validation is offline/mock protocol coverage, not live Google account acceptance. Sources: [Google Cloud](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai), [Tool Search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool), [mid-conversation controls](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages), [official Vertex SDK](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/lib/vertex/_client.py).
