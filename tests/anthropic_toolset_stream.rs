#[path = "support/wire_api.rs"]
mod wire_api;

use lingxi_llm_client::protocol::{ContentBlock, StreamEvent};
use lingxi_llm_client::{
    AnthropicMessagesCodec, CodecContext, HttpResponse, RequestMode, WireCodec,
};
use serde_json::json;

const MODEL: &str = "claude-opus-5-5";

fn profile() -> lingxi_llm_client::protocol::ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "anthropic",
        "profile_name": "anthropic",
        "base_url": "https://api.anthropic.com",
        "protocol": "anthropic_messages",
        "auth": "none",
        "models": [{"display_model": MODEL, "request_model": MODEL, "billing_model": MODEL}]
    }))
    .unwrap()
}

#[test]
fn response_decode_preserves_browser_toolset_namespace() {
    let response = HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": MODEL,
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use",
                "id": "toolu_browser",
                "name": "screenshot",
                "input": {"tab_id": "tab-1"},
                "toolset_name": "browser"
            }]
        }))
        .unwrap()
        .into(),
    };
    let context = CodecContext::new(&profile(), MODEL, RequestMode::Complete);
    let decoded = AnthropicMessagesCodec
        .decode_response(&response, &context)
        .unwrap();
    assert!(matches!(
        decoded.message.content.as_slice(),
        [ContentBlock::ToolUse {
            toolset_name: Some(toolset_name),
            name,
            ..
        }] if toolset_name == "browser" && name == "screenshot"
    ));
    let (id, toolset_name, name, input) = decoded.message.tool_uses().next().unwrap();
    assert_eq!(id.as_str(), "toolu_browser");
    assert_eq!(toolset_name, Some("browser"));
    assert_eq!(name, "screenshot");
    assert_eq!(input["tab_id"], "tab-1");
}

#[test]
fn stream_carries_distinct_toolset_names_on_complete_tool_input_deltas() {
    let profile = profile();
    let context = CodecContext::new(&profile, MODEL, RequestMode::Stream);
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&context);
    let frames = [
        json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":MODEL,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_browser","name":"screenshot","input":{},"toolset_name":"browser"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_computer","name":"screenshot","input":{},"toolset_name":"computer"}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":1}}),
        json!({"type":"message_stop"}),
    ];
    let mut events = Vec::new();
    for frame in frames {
        let encoded = serde_json::to_vec(&frame).unwrap();
        events.extend(wire_api::decode_frame(&mut *decoder, &encoded).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());

    let calls = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolCallDelta {
                toolset_name,
                name,
                arguments_fragment,
                ..
            } => Some((
                toolset_name.as_deref(),
                name.as_str(),
                arguments_fragment.as_str(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 4);
    assert_eq!(calls[0], (Some("browser"), "screenshot", ""));
    assert_eq!(calls[1], (Some("browser"), "screenshot", "{}"));
    assert_eq!(calls[2], (Some("computer"), "screenshot", ""));
    assert_eq!(calls[3], (Some("computer"), "screenshot", "{}"));
}
