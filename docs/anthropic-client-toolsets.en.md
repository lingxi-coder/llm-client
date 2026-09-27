# Anthropic Browser / Computer client toolsets

Declare stable `browser_toolset_20260801` / `computer_toolset_20260801` through `ChatRequest.anthropic_client_toolsets`. The host owns permission checks, execution and results. These declarations are separate from `hosted_tools`; this library does not open browsers or control a desktop.

```rust
use lingxi_llm_client::protocol::{
    AnthropicBrowserMember, AnthropicBrowserToolsetConfig, AnthropicClientToolConfig,
    AnthropicClientToolset, ChatRequest,
};
# fn configure(request: &mut ChatRequest) {
let mut browser = AnthropicBrowserToolsetConfig::default();
browser.configs.insert(AnthropicBrowserMember::JavascriptExec,
    AnthropicClientToolConfig { enabled: Some(true), defer_loading: None });
request.anthropic_client_toolsets.push(AnthropicClientToolset::Browser(browser));
# }
```

Entries have no `name`. Member configs accept only `enabled` / `defer_loading`. Browser has 31 members, with `javascript_exec`, `file_upload`, `read_console` and `read_network` disabled by default. All 17 Computer members default to enabled. Sparse config preserves provider defaults.

Both toolsets may coexist, and custom tools may share a member name. Dispatch using `(toolset_name, name)`. `ToolUse`, `ToolResult`, `StreamEvent::ToolCallDelta`, and `ConversationMessage::tool_uses()` preserve the namespace. Results must echo the call's namespace; ordinary tools use `None`. Rust literals require the new field; there is no legacy API compatibility branch.

```rust
use lingxi_llm_client::protocol::{ContentBlock, ConversationMessage};
# fn results(message: &ConversationMessage) -> Vec<ContentBlock> {
message.content.iter().filter_map(|block| {
    let ContentBlock::ToolUse { id, toolset_name, name, .. } = block else { return None };
    // Dispatch using both toolset_name and name in the host application.
    Some(ContentBlock::ToolResult {
        tool_use_id: id.clone(),
        toolset_name: toolset_name.clone(),
        content: format!("Host execution unavailable for {toolset_name:?}/{name}"),
        is_error: true,
        blocks: None,
    })
}).collect()
# }
```

The example returns an error without executing anything. Successful `new_tab`, `switch_tab`, `close_tab` and `list_tabs` return exactly one native `browser_state`. Other Browser results may carry text/image and at most one state block; errors must not include state. Preserve state through `ToolResult.blocks`. Computer results accept text/image only.

Omit `allowed_callers` or set it to `[Direct]`. Programmatic callers, strict, input_examples and entry-level defer_loading are unsupported. All enabled members must agree on defer_loading. A deferred set requires tool search and cannot carry a cache marker. Toolset markers share the request's four-marker and TTL-order validation. All-disabled/unknown members, duplicate sets, custom tools named browser/computer, forced tool_choice naming a set/member, and the legacy fine-grained streaming beta are rejected.

This adapter supports first-party Anthropic Messages and Vertex Claude on the documented Fable 5/5.1, Mythos 5/5.1, Opus 4.8/5/5.5 and Sonnet 5 models without a beta header. Vertex uses the same stable toolset IDs and namespace rules. Foundry and Bedrock stable toolsets are outside this interface's supported routes. Other codecs reject declarations and historical namespaces rather than dropping them. Local token estimates explicitly omit provider-defined member schemas instead of treating them as zero.

Offline/mock tests do not establish account availability or host browser/desktop behavior. Sources: [Tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference), [Browser](https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-use-tool), [Computer](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool).

The host executor owns member output conventions: successful screenshot/zoom should return an image; other non-tab-management members should return text, with optional additional images. The guide explicitly says these conventions are not API-enforced, so the client does not impose them as wire restrictions. Native block types, tab-management shapes and browser_state structural constraints are still validated. Download size_bytes accepts null or a non-negative integer.
