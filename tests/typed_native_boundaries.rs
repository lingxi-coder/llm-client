use lingxi_llm_client::{
    codecs::{
        anthropic::AnthropicMessagesCodec, gemini::GeminiCodec, openai::chat::OpenAiChatCodec,
    },
    execution_safety::{request_replay_safety, RequestReplaySafety},
    protocol::*,
    replay::{ReplayContext, ReplayPolicy},
    CodecContext, EncodeRequest, RequestMode, WireCodec,
};
use serde_json::json;

fn unknown_block() -> ContentBlock {
    ContentBlock::Native {
        value: NativeExtension::new("future.client_call.v1", json!({"call_id":"call_a"})).unwrap(),
    }
}

#[test]
fn typed_native_content_requires_explicit_replay_and_retry_decisions() {
    let mut request = ChatRequest::new("m");
    request
        .messages
        .push(ConversationMessage::assistant(vec![unknown_block()]));
    assert_eq!(
        request_replay_safety(&request),
        RequestReplaySafety::StatefulOrUnknown
    );
    for family in [
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::AnthropicMessages,
        ProtocolFamily::GeminiGenerateContent,
    ] {
        let replay = ReplayContext::for_message(&request.messages[0].content, family);
        assert!(replay
            .normalize(&unknown_block(), None, ReplayPolicy::Reject)
            .is_err());
        assert_eq!(
            replay
                .normalize(&unknown_block(), None, ReplayPolicy::DropIncompatible)
                .unwrap(),
            None
        );
    }
}

#[test]
fn other_codecs_never_drop_or_reinterpret_typed_native_content() {
    let codecs: Vec<Box<dyn WireCodec>> = vec![
        Box::new(OpenAiChatCodec),
        Box::new(AnthropicMessagesCodec),
        Box::new(GeminiCodec),
    ];
    for codec in codecs {
        let profile: ProviderProfile = serde_json::from_value(json!({
            "profile_name":"test", "provider_id":"test", "protocol":codec.family(),
            "base_url":"https://example.test", "auth":"none"
        }))
        .unwrap();
        let context = CodecContext::new(&profile, "m", RequestMode::Complete);
        for content in [true, false] {
            let mut request = ChatRequest::new("m");
            if content {
                request
                    .messages
                    .push(ConversationMessage::assistant(vec![unknown_block()]));
            } else {
                request.set_openai_computer_tool(Some(Default::default()));
                request
                    .messages
                    .push(ConversationMessage::user_text("Inspect the current screen"));
            }
            assert!(
                matches!(
                    codec.validate_request(&request, &context),
                    Err(LlmError::UnsupportedCapability { .. })
                ),
                "{:?}",
                codec.family()
            );
            assert!(
                matches!(
                    codec.encode_request(EncodeRequest::new(&request), &context),
                    Err(LlmError::UnsupportedCapability { .. })
                ),
                "{:?}",
                codec.family()
            );
        }
    }
}

#[test]
fn native_stream_assembly_preserves_payload_without_inventing_tool_calls() {
    use lingxi_llm_client::stream_assembly::StreamAccumulator;
    let ContentBlock::Native { value } = unknown_block() else {
        unreachable!()
    };
    let mut assembly = StreamAccumulator::new();
    assembly.observe(&StreamEvent::Native {
        block: 7,
        value: value.clone(),
    });
    assert!(!assembly.snapshot().terminal);
    assert!(!assembly.snapshot().indexed_content.contains_key(&7));
    assert!(matches!(
        assembly.snapshot().events.last(),
        Some(StreamEvent::Native { block: 7, .. })
    ));
    assembly.observe(&StreamEvent::End {
        stop_reason: StopReason::ToolUse,
        usage: UsageReport::default(),
        inference: Default::default(),
    });
    let result = assembly.finish().unwrap();
    assert!(result.terminal);
    assert_eq!(result.response.message.content, vec![unknown_block()]);
    assert!(result
        .response
        .message
        .content
        .iter()
        .all(|block| !matches!(block, ContentBlock::ToolUse { .. })));
}

#[test]
fn generated_and_raw_computer_calls_cannot_be_replayed_as_input() {
    use lingxi_llm_client::providers::openai::computer::OpenAiComputerCall;
    let item = json!({"type":"computer_call", "id":"cci_1", "call_id":"call_1", "status":"completed",
        "pending_safety_checks":[],
        "actions":[{"type":"screenshot"}]});
    let call = OpenAiComputerCall::from_response_item(&item)
        .unwrap()
        .into_content_block()
        .unwrap();
    for block in [
        call,
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: item,
        },
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: json!({"type":"computer_call_output", "call_id":"call_1"}),
        },
    ] {
        let replay = ReplayContext::for_message(
            std::slice::from_ref(&block),
            ProtocolFamily::OpenAiResponses,
        );
        assert!(replay
            .normalize(&block, None, ReplayPolicy::Reject)
            .is_err());
    }
}

#[test]
fn truncated_or_refused_terminal_streams_do_not_materialize_native_calls() {
    use lingxi_llm_client::stream_assembly::StreamAccumulator;
    for reason in [
        StopReason::MaxTokens,
        StopReason::Refusal,
        StopReason::Other("stream_interrupted".into()),
    ] {
        let ContentBlock::Native { value } = unknown_block() else {
            unreachable!()
        };
        let mut assembly = StreamAccumulator::new();
        assembly.observe(&StreamEvent::Native { block: 0, value });
        assembly.observe(&StreamEvent::End {
            stop_reason: reason,
            usage: UsageReport::default(),
            inference: Default::default(),
        });
        assert!(assembly.snapshot().terminal);
        assert!(assembly.snapshot().indexed_content.is_empty());
        assert!(assembly
            .snapshot()
            .events
            .iter()
            .any(|event| matches!(event, StreamEvent::Native { .. })));
    }
}
