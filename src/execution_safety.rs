//! Conservative execution-safety facts for callers that own dispatch and retries.
//!
//! This module never sends requests or chooses retries. A stateless input can
//! still be unsafe to repeat after a particular failure; callers must also apply
//! their transport, settlement, and retry policies.

use crate::protocol::{ChatRequest, ContentBlock, JsonPayload, NativeExtension};
#[cfg(test)]
use serde_json::Value;

/// Whether canonical input may refer to provider work or unrecognized state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestReplaySafety {
    /// No known provider execution or opaque state is present in the input.
    Stateless,
    /// Repetition requires an explicit caller decision because work or state
    /// may already exist at the provider.
    StatefulOrUnknown,
}

/// Classify canonical request input without authorizing a retry or failover.
/// Hosted tools and unknown extensions are conservative, including future ones.
#[must_use]
pub fn request_replay_safety(request: &ChatRequest) -> RequestReplaySafety {
    let stateful = request.has_response_continuation()
        || !request.hosted_tools.is_empty()
        || request
            .native_options
            .iter()
            .any(|extension| !stateless_options(extension))
        || request.messages.iter().any(|message| {
            !message.native_options.is_empty()
                || message.content.iter().any(|block| match block {
                    ContentBlock::Native { .. } => true,
                    ContentBlock::ToolUse {
                        caller,
                        toolset_name,
                        ..
                    } => toolset_name.is_some() || caller.as_ref().is_some_and(indirect_caller),
                    ContentBlock::ToolResult { toolset_name, .. } => toolset_name.is_some(),
                    ContentBlock::ProviderContent { value, .. } => {
                        native_content_may_execute(value)
                    }
                    _ => false,
                })
        });
    if stateful {
        RequestReplaySafety::StatefulOrUnknown
    } else {
        RequestReplaySafety::Stateless
    }
}

fn stateless_options(extension: &NativeExtension) -> bool {
    use crate::providers::anthropic::native::AnthropicRequestOptions;
    // Only host-executed client tool declarations are understood here. A new
    // SDK field needs review before it becomes stateless by deserialization.
    extension
        .data()
        .has_only_fields_at("", &["client_toolsets"])
        && extension.decode::<AnthropicRequestOptions>().is_ok()
}

fn indirect_caller(caller: &JsonPayload) -> bool {
    !caller.is_root_null() && caller.string_at("/type").as_deref() != Some("direct")
}

fn native_content_may_execute(value: &JsonPayload) -> bool {
    match value.string_at("/type").as_deref() {
        Some("text" | "thinking" | "redacted_thinking" | "reasoning" | "chat_reasoning") => false,
        Some("tool_use") => {
            nonnull_field(value, "/toolset_name")
                || (nonnull_field(value, "/caller") && !value.string_is("/caller/type", "direct"))
        }
        Some("tool_result") => nonnull_field(value, "/toolset_name"),
        _ => true,
    }
}

fn nonnull_field(value: &JsonPayload, pointer: &str) -> bool {
    value.raw_source_at(pointer).is_some()
        && value.kind_at(pointer) != Some(crate::response_json::FlatJsonKind::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        ConversationMessage, HostedTool, MessageRole, NativeType, ProtocolFamily, WebSearchConfig,
    };
    use crate::providers::anthropic::native::AnthropicRequestOptions;
    use serde_json::json;

    fn request(block: ContentBlock) -> ChatRequest {
        let mut request = ChatRequest::new("model");
        request.messages.push(ConversationMessage {
            role: MessageRole::Assistant,
            content: vec![block],
            native_options: vec![],
        });
        request
    }
    fn native(value: Value) -> ContentBlock {
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: JsonPayload::from_generated_value(&value).unwrap(),
        }
    }

    #[test]
    fn ordinary_reasoning_and_direct_host_tools_are_stateless() {
        for value in [
            json!({"type":"text","text":"cited","citations":[{}]}),
            json!({"type":"reasoning","encrypted_content":"opaque"}),
            json!({"type":"chat_reasoning","reasoning_content":"thought"}),
            json!({"type":"tool_use","caller":{"type":"direct"}}),
        ] {
            assert_eq!(
                request_replay_safety(&request(native(value))),
                RequestReplaySafety::Stateless
            );
        }
        let direct: ContentBlock = serde_json::from_value(json!({"type":"tool_use","id":"call","name":"lookup","input":{},"caller":{"type":"direct"}})).unwrap();
        assert_eq!(
            request_replay_safety(&request(direct)),
            RequestReplaySafety::Stateless
        );
    }

    #[test]
    fn unknown_native_state_callers_and_toolsets_require_explicit_decision() {
        for value in [
            json!({"type":"server_tool_use","name":"web_fetch"}),
            json!({"type":"mcp_tool_result"}),
            json!({"executableCode":{"code":"print(1)"}}),
            json!({"type":"future_state"}),
            json!({"type":"tool_use","toolset_name":"browser"}),
        ] {
            assert_eq!(
                request_replay_safety(&request(native(value))),
                RequestReplaySafety::StatefulOrUnknown
            );
        }
        for block in [
            json!({"type":"tool_use","id":"c","name":"run","input":{},"caller":{"type":"code_execution_20260120"}}),
            json!({"type":"tool_use","id":"c","name":"click","input":{},"toolset_name":"browser"}),
            json!({"type":"tool_result","tool_use_id":"c","content":"done","is_error":false,"toolset_name":"browser"}),
        ] {
            assert_eq!(
                request_replay_safety(&request(serde_json::from_value(block).unwrap())),
                RequestReplaySafety::StatefulOrUnknown
            );
        }
    }

    #[test]
    fn legacy_continuations_hosted_tools_and_message_extensions_are_not_stateless() {
        let mut req = ChatRequest::new("model");
        req.controls.responses.previous_response_id = Some("previous".into());
        assert_eq!(
            request_replay_safety(&req),
            RequestReplaySafety::StatefulOrUnknown
        );
        req.controls.responses.previous_response_id = None;
        req.hosted_tools
            .push(HostedTool::WebSearch(WebSearchConfig::default()));
        assert_eq!(
            request_replay_safety(&req),
            RequestReplaySafety::StatefulOrUnknown
        );
        req = request(native(json!({"type":"text","text":"x"})));
        req.messages[0].native_options.push(
            NativeExtension::new(
                "future.message.v1",
                JsonPayload::from_generated_value(&json!({"state":"id"})).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(
            request_replay_safety(&req),
            RequestReplaySafety::StatefulOrUnknown
        );
    }

    #[test]
    fn only_recognized_client_toolset_options_are_stateless() {
        let mut req = ChatRequest::new("model");
        req.native_options.push(
            NativeExtension::new(
                AnthropicRequestOptions::FORMAT,
                JsonPayload::from_generated_value(&json!({"client_toolsets":[]})).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(request_replay_safety(&req), RequestReplaySafety::Stateless);
        for (format, data) in [
            ("future.state.v1", json!({"id":"state"})),
            (
                AnthropicRequestOptions::FORMAT,
                json!({"client_toolsets":[],"container":"old"}),
            ),
            (
                AnthropicRequestOptions::FORMAT,
                json!({"client_toolsets":42}),
            ),
        ] {
            req.native_options = vec![NativeExtension::new(
                format,
                JsonPayload::from_generated_value(&data).unwrap(),
            )
            .unwrap()];
            assert_eq!(
                request_replay_safety(&req),
                RequestReplaySafety::StatefulOrUnknown
            );
        }
    }
}
