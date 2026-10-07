use lingxi_llm_client::protocol::*;
use lingxi_llm_client::providers::{
    anthropic::computer::SKIPPED_COMPUTER_ACTION, openai::computer::OpenAiComputerCall,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, CodecContext, EncodeRequest, RequestMode, WireCodec,
};
use serde_json::{json, Value};
fn frame() -> ComputerFrame {
    ComputerFrame {
        width: 1024,
        height: 768,
        geometry_version: "full-display-v1".into(),
    }
}
fn continuation() -> ContinuationRef {
    serde_json::from_value(json!({"protocol":"open_ai_responses","response_id":"resp1","provider_id":"openai","profile_name":"openai","endpoint_fingerprint":"endpoint","account_scope":"account","request_model":"gpt-6-astra"})).unwrap()
}
fn member(id: &str, name: &str, input: Value) -> ContentBlock {
    serde_json::from_value(
        json!({"type":"tool_use","id":id,"name":name,"input":input,"toolset_name":"computer"}),
    )
    .unwrap()
}
fn result(
    index: usize,
    status: NativeExecutionStatus,
    blocks: Option<Vec<Value>>,
) -> NativeComputerResult {
    NativeComputerResult {
        operation_index: index,
        status,
        content: "actual final result".into(),
        blocks,
    }
}
fn receipt(results: Vec<NativeComputerResult>) -> ComputerReceiptInput {
    ComputerReceiptInput {
        results,
        acknowledged_safety_checks: vec![],
    }
}
#[test]
fn openai_retains_actions_modifiers_geometry_safety_and_requires_explicit_observation() {
    let original=OpenAiComputerCall::from_response_item(&json!({"type":"computer_call","id":"item1","call_id":"call1","status":"completed","pending_safety_checks":[{"id":"check1","code":"confirm"}],"actions":[{"type":"click","button":"wheel","x":10,"y":20,"keys":["SHIFT"]},{"type":"drag","path":[{"x":1,"y":2},{"x":3,"y":4},{"x":5,"y":6}]},{"type":"wait"}]})).unwrap();
    let block = original.into_content_block().unwrap();
    let calls = decode_computer_calls(
        NativeComputerProvider::OpenAi,
        &[block.clone()],
        Some(&continuation()),
        &frame(),
    )
    .unwrap();
    let call = &calls[0];
    assert_eq!(call.operations.len(), 4);
    assert_eq!(call.context.action_count, 4);
    assert!(
        matches!(&call.operations[0],ComputerOperation::Click{button:ComputerMouseButton::Middle,modifiers,..} if modifiers==&["SHIFT"])
    );
    assert!(matches!(&call.operations[1],ComputerOperation::Drag{path,..} if path.len()==3));
    assert_eq!(
        call.operations[2],
        ComputerOperation::Wait {
            duration_seconds: 2.0
        }
    );
    assert_eq!(call.operations[3], ComputerOperation::Screenshot);
    assert_eq!(call.context.pending_safety_checks[0]["id"], "check1");
    assert!(
        decode_computer_calls(NativeComputerProvider::OpenAi, &[block], None, &frame()).is_err()
    );
    let input = receipt(
        (0..4)
            .map(|i| result(i, NativeExecutionStatus::Succeeded, None))
            .collect(),
    );
    assert!(encode_computer_receipt(call, &input).is_err());
    let mut input = input;
    input.results[3].blocks = Some(vec![
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}),
    ]);
    let output = encode_computer_receipt(call, &input).unwrap();
    let value = match output {
        ContentBlock::Native { value } => value,
        _ => panic!(),
    };
    assert_eq!(
        value.data()["output"]["image_url"],
        "data:image/png;base64,AA=="
    );
    assert!(value.data().get("acknowledged_safety_checks").is_none());
    input.results[1].status = NativeExecutionStatus::Failed;
    input.results[2].status = NativeExecutionStatus::Skipped;
    input.results[3].status = NativeExecutionStatus::Skipped;
    assert!(encode_computer_receipt(call, &input).is_err());
}
#[test]
fn claude_all_members_preserve_cursor_wheel_repeat_and_duration_semantics() {
    let members = vec![
        member("1", "screenshot", json!({})),
        member("2", "zoom", json!({"region":[0,0,1024,768]})),
        member("3", "left_click", json!({"text":"ctrl+shift"})),
        member("4", "right_click", json!({"coordinate":[1,2]})),
        member("5", "middle_click", json!({})),
        member("6", "double_click", json!({})),
        member("7", "triple_click", json!({})),
        member(
            "8",
            "left_click_drag",
            json!({"start_coordinate":[1,2],"coordinate":[3,4]}),
        ),
        member("9", "mouse_move", json!({"coordinate":[1,2]})),
        member("10", "left_mouse_down", json!({})),
        member("11", "left_mouse_up", json!({})),
        member("12", "cursor_position", json!({})),
        member(
            "13",
            "scroll",
            json!({"scroll_direction":"down","scroll_amount":3}),
        ),
        member("14", "type", json!({"text":"hello"})),
        member("15", "key", json!({"text":"ctrl+s","repeat":4})),
        member("16", "hold_key", json!({"text":"shift","duration":300})),
        member("17", "wait", json!({"duration":0.5})),
    ];
    let calls =
        decode_computer_calls(NativeComputerProvider::Anthropic, &members, None, &frame()).unwrap();
    assert_eq!(calls.len(), 17);
    assert!(
        matches!(&calls[2].operations[0],ComputerOperation::Click{target:ComputerTarget::CurrentCursor,modifiers,..} if modifiers==&["ctrl","shift"])
    );
    assert!(matches!(
        &calls[12].operations[0],
        ComputerOperation::ScrollWheel {
            target: ComputerTarget::CurrentCursor,
            amount: 3,
            direction: ComputerScrollDirection::Down,
            ..
        }
    ));
    assert_eq!(calls[14].operations.len(), 4);
    assert!(calls[14]
        .operations
        .iter()
        .all(|op| matches!(op,ComputerOperation::Key{keys} if keys==&["ctrl","s"])));
    assert_eq!(
        calls[15].operations[0],
        ComputerOperation::HoldKey {
            keys: vec!["shift".into()],
            duration_seconds: 300.0
        }
    );
}
#[test]
fn decode_validates_entire_batch_and_receipts_enforce_stopping_and_hook_output() {
    let valid = member("1", "left_click", json!({"coordinate":[1,2]}));
    for invalid in [
        member(
            "2",
            "scroll",
            json!({"scroll_direction":"down","scroll_amount":1,"pixel_delta":[0,1]}),
        ),
        member("2", "mouse_move", json!({"coordinate":[1024,0]})),
        member("2", "key", json!({"text":"ctrl++s"})),
        member("2", "wait", json!({"duration":301})),
    ] {
        assert!(decode_computer_calls(
            NativeComputerProvider::Anthropic,
            &[valid.clone(), invalid],
            None,
            &frame()
        )
        .is_err());
    }
    let calls = decode_computer_calls(
        NativeComputerProvider::Anthropic,
        &[member("1", "key", json!({"text":"Tab","repeat":3}))],
        None,
        &frame(),
    )
    .unwrap();
    let bad = receipt(vec![
        result(0, NativeExecutionStatus::Failed, None),
        result(1, NativeExecutionStatus::Succeeded, None),
        result(2, NativeExecutionStatus::Skipped, None),
    ]);
    assert!(encode_computer_receipt(&calls[0], &bad).is_err());
    let skipped = receipt(
        (0..3)
            .map(|i| result(i, NativeExecutionStatus::Skipped, None))
            .collect(),
    );
    assert!(
        matches!(encode_computer_receipt(&calls[0],&skipped).unwrap(),ContentBlock::ToolResult{is_error:Some(true),content,toolset_name:Some(toolset),..} if content==SKIPPED_COMPUTER_ACTION && toolset=="computer")
    );
    let shot = decode_computer_calls(
        NativeComputerProvider::Anthropic,
        &[member("shot", "screenshot", json!({}))],
        None,
        &frame(),
    )
    .unwrap();
    assert!(encode_computer_receipt(
        &shot[0],
        &receipt(vec![result(0, NativeExecutionStatus::Succeeded, None)])
    )
    .is_err());
    let image =
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}});
    assert!(encode_computer_receipt(
        &shot[0],
        &receipt(vec![result(
            0,
            NativeExecutionStatus::Succeeded,
            Some(vec![image])
        )])
    )
    .is_ok());
}
#[test]
fn claude_old_computer_member_history_encodes_after_switch_to_function_round() {
    let block = member("call1", "left_click", json!({"coordinate":[1,2]}));
    let call = decode_computer_calls(
        NativeComputerProvider::Anthropic,
        &[block.clone()],
        None,
        &frame(),
    )
    .unwrap();
    let output = encode_computer_receipt(
        &call[0],
        &receipt(vec![result(0, NativeExecutionStatus::Succeeded, None)]),
    )
    .unwrap();
    let mut request = ChatRequest::new("claude-opus-5-5");
    request.tools.push(serde_json::from_value(json!({"name":"computer","description":"Manage access","input_schema":{"type":"object","properties":{}}})).unwrap());
    request.messages = vec![
        ConversationMessage {
            role: MessageRole::Assistant,
            content: vec![block],
            native_options: vec![],
        },
        ConversationMessage {
            role: MessageRole::User,
            content: vec![output],
            native_options: vec![],
        },
    ];
    let profile:ProviderProfile=serde_json::from_value(json!({"provider_id":"anthropic","profile_name":"anthropic","base_url":"https://api.anthropic.com","protocol":"anthropic_messages","auth":"none","regions":["international"],"models":[{"display_model":"claude-opus-5-5","request_model":"claude-opus-5-5","billing_model":"claude-opus-5-5"}]})).unwrap();
    let wire = AnthropicMessagesCodec
        .encode_request(
            EncodeRequest::new(&request),
            &CodecContext::new(&profile, "claude-opus-5-5", RequestMode::Complete),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    assert_eq!(body["tools"][0]["name"], "computer");
    assert_eq!(
        body["messages"][1]["content"][0]["toolset_name"],
        "computer"
    );
}

#[test]
fn declarations_are_capability_filtered_and_round_exclusive() {
    let mut request = ChatRequest::new("claude-opus-5-5");
    let caps = ComputerCapabilities {
        operations: vec![
            ComputerOperationKind::Screenshot,
            ComputerOperationKind::Click,
        ],
    };
    declare_computer_tool(
        &mut request,
        NativeComputerProvider::Anthropic,
        &caps,
        &frame(),
    )
    .unwrap();
    let toolset = &request.anthropic_client_toolsets()[0];
    assert_eq!(toolset.member_enabled("left_click"), Some(true));
    assert_eq!(toolset.member_enabled("zoom"), Some(false));
    assert_eq!(toolset.member_enabled("scroll"), Some(false));
    request.tools.push(serde_json::from_value(json!({"name":"computer","description":"Manage access","input_schema":{"type":"object","properties":{}}})).unwrap());
    assert!(declare_computer_tool(
        &mut request,
        NativeComputerProvider::Anthropic,
        &caps,
        &frame()
    )
    .is_err());
    let mut openai = ChatRequest::new("gpt-6-astra");
    assert!(
        declare_computer_tool(&mut openai, NativeComputerProvider::OpenAi, &caps, &frame())
            .is_err()
    );
    assert!(openai.openai_computer_tool().is_none());
}

#[test]
fn duplicate_call_ids_are_rejected_before_any_normalized_call_is_returned() {
    let blocks = vec![
        member("same", "left_click", json!({})),
        member("same", "screenshot", json!({})),
    ];
    assert!(
        decode_computer_calls(NativeComputerProvider::Anthropic, &blocks, None, &frame()).is_err()
    );
}

#[test]
fn history_call_identity_is_namespaced_and_never_matches_ordinary_member_functions() {
    let native = member("native-id", "screenshot", json!({}));
    assert_eq!(computer_call_id(&native), Some("native-id"));
    let ordinary: ContentBlock = serde_json::from_value(
        json!({"type":"tool_use","id":"ordinary-id","name":"screenshot","input":{}}),
    )
    .unwrap();
    assert_eq!(computer_call_id(&ordinary), None);
    let output:ContentBlock=serde_json::from_value(json!({"type":"tool_result","tool_use_id":"native-id","content":"ok","is_error":false,"toolset_name":"computer"})).unwrap();
    assert_eq!(computer_call_id(&output), None);
}
