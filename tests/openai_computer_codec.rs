use lingxi_llm_client::{
    codecs::{
        openai::responses::OpenAiResponsesCodec, CodecContext, EncodeRequest, RequestMode,
        StreamDecoder, WireCodec,
    },
    protocol::{
        ChatRequest, ChatResponse, ContentBlock, ContinuationRef, ConversationMessage, LlmError,
        MessageRole, NativeExtension, ProtocolFamily, ProviderId, ProviderProfile, ResponseId,
        StopReason, StreamEvent,
    },
    providers::openai::computer::{
        ComputerAction, ComputerCallStatus, ComputerSafetyCheck, OpenAiComputerCall,
        OpenAiComputerCallOutput, OpenAiComputerToolConfig,
    },
    HttpResponse,
};
use serde_json::{json, Value};

#[path = "support/wire_api.rs"]
mod wire_api;

const MODEL: &str = "gpt-6-astra";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai",
        "profile_name":"openai",
        "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_responses",
        "auth":"none",
        "extra":{"supports_previous_response_id":true},
        "models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]
    }))
    .unwrap()
}

fn context(profile: &ProviderProfile, mode: RequestMode) -> CodecContext {
    CodecContext::new(profile, MODEL, mode)
}

fn request() -> ChatRequest {
    let mut request = ChatRequest::new(MODEL);
    request
        .messages
        .push(ConversationMessage::user_text("continue"));
    request
}

fn call_item(call_id: &str) -> Value {
    json!({
        "type":"computer_call",
        "id":format!("cci_{call_id}"),
        "call_id":call_id,
        "pending_safety_checks":[{"id":"safety_1","code":"confirm","message":"Review this action"}],
        "status":"completed",
        "actions":[{"type":"screenshot"}]
    })
}

fn decode(body: Value) -> Result<ChatResponse, LlmError> {
    let profile = profile();
    OpenAiResponsesCodec.decode_response(
        &HttpResponse {
            status: 200,
            headers: vec![],
            body: serde_json::to_vec(&body).unwrap().into(),
        },
        &context(&profile, RequestMode::Complete),
    )
}

fn encode(request: &ChatRequest, profile: &ProviderProfile) -> Result<Value, LlmError> {
    let http = OpenAiResponsesCodec.encode_request(
        EncodeRequest::new(request),
        &context(profile, RequestMode::Complete),
    )?;
    Ok(serde_json::from_slice(&http.body).unwrap())
}

fn push_stream_frame(
    decoder: &mut dyn StreamDecoder,
    frame: &Value,
) -> Vec<Result<StreamEvent, LlmError>> {
    let data = format!("data: {frame}\n\n");
    decoder.push_bytes(data.as_bytes())
}

fn computer_stream_decoder(profile: &ProviderProfile) -> Box<dyn StreamDecoder> {
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&context(profile, RequestMode::Stream));
    let created = json!({"type":"response.created","response":{"id":"resp_stream","model":MODEL}});
    assert!(push_stream_frame(&mut *decoder, &created)
        .iter()
        .all(Result::is_ok));
    decoder
}

#[test]
fn completed_call_is_one_typed_native_block_and_preserves_order_and_safety_checks() {
    let response = decode(json!({
        "id":"resp-computer",
        "model":MODEL,
        "status":"completed",
        "output":[call_item("call_1")],
        "usage":{"input_tokens":12,"output_tokens":4}
    }))
    .unwrap();

    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.message.content.len(), 1);
    let call = OpenAiComputerCall::from_content_block(&response.message.content[0]).unwrap();
    assert_eq!(call.status, ComputerCallStatus::Completed);
    assert_eq!(call.actions, vec![ComputerAction::Screenshot]);
    assert_eq!(call.pending_safety_checks[0].id, "safety_1");
    assert!(response.usage.usage.is_some());
}

#[test]
fn completed_nonstream_call_requires_explicit_safety_checks_array() {
    let mut item = call_item("call_empty_checks");
    item["pending_safety_checks"] = json!([]);
    let response = decode(json!({
        "id":"resp_empty_checks", "status":"completed", "output":[item.clone()]
    }))
    .unwrap();
    let ContentBlock::Native { value } = &response.message.content[0] else {
        panic!("completed computer call must be typed native content");
    };
    assert_eq!(value.data()["pending_safety_checks"], json!([]));
    assert!(OpenAiComputerCall::from_extension(value).is_ok());

    item.as_object_mut()
        .unwrap()
        .remove("pending_safety_checks");
    assert!(matches!(
        decode(json!({"id":"resp_missing_checks", "status":"completed", "output":[item]})),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn completed_nonstream_computer_call_requires_response_id() {
    for id in [None, Some("")] {
        let mut body = json!({"status":"completed","output":[call_item("call_no_response_id")]});
        if let Some(id) = id {
            body["id"] = json!(id);
        }
        assert!(matches!(decode(body), Err(LlmError::InvalidRequest { .. })));
    }
}

#[test]
fn completed_nonstream_computer_call_requires_item_id() {
    for invalid_id in [None, Some(json!("")), Some(Value::Null)] {
        let mut item = call_item("call_no_item_id");
        if let Some(id) = invalid_id {
            item["id"] = id;
        } else {
            item.as_object_mut().unwrap().remove("id");
        }
        assert!(matches!(
            decode(json!({"id":"resp_item_id", "status":"completed", "output":[item]})),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}

#[test]
fn computer_tool_config_and_matching_output_use_responses_items_and_scoped_continuation() {
    let profile = profile();
    let source_call = OpenAiComputerCall::from_response_item(&call_item("call_2")).unwrap();
    let output: OpenAiComputerCallOutput = serde_json::from_value(json!({
        "type":"computer_call_output",
        "call_id":"call_2",
        "output":{"type":"computer_screenshot","image_url":"https://example.test/screenshot.png"}
    }))
    .unwrap();
    let output = output.into_content_block_for(&source_call).unwrap();
    let mut request = request();
    request.set_openai_computer_tool(Some(OpenAiComputerToolConfig::default()));
    request.continuation = Some(ContinuationRef {
        response_id: ResponseId::new("resp-previous"),
        provider_id: ProviderId::new("openai"),
        profile_name: "openai".into(),
        endpoint_fingerprint: "endpoint".into(),
        account_scope: "test-account".into(),
        request_model: MODEL.into(),
        workspace_id: None,
    });
    request.messages = vec![ConversationMessage {
        role: MessageRole::User,
        content: vec![output],
        native_options: vec![],
    }];

    let body = encode(&request, &profile).unwrap();
    assert_eq!(body["previous_response_id"], "resp-previous");
    assert_eq!(body["tools"], json!([{"type":"computer"}]));
    assert_eq!(body["input"][0]["type"], "computer_call_output");
    assert_eq!(body["input"][0]["call_id"], "call_2");
    assert_eq!(
        body["input"][0]["output"]["image_url"],
        "https://example.test/screenshot.png"
    );
    assert!(body["input"][0].get("acknowledged_safety_checks").is_none());
    assert!(body["input"][0].get("function_call_output").is_none());
}

#[test]
fn duplicate_function_and_computer_call_ids_are_rejected_in_nonstream_response() {
    let error = decode(json!({
        "status":"completed",
        "output":[
            {"type":"function_call","call_id":"same_call","name":"lookup","arguments":"{}"},
            call_item("same_call")
        ]
    }))
    .unwrap_err();
    assert!(matches!(error, LlmError::InvalidRequest { .. }));
}

#[test]
fn computer_call_requires_consistent_terminal_response_status() {
    for status in [None, Some("in_progress")] {
        let mut body = json!({"output":[call_item("call_status")]});
        if let Some(status) = status {
            body["status"] = json!(status);
        }
        assert!(matches!(decode(body), Err(LlmError::InvalidRequest { .. })));
    }
}

#[test]
fn incomplete_call_is_preserved_as_non_executable_provider_content() {
    let mut item = call_item("call_incomplete");
    item["status"] = json!("incomplete");
    let response = decode(json!({
        "status":"incomplete",
        "incomplete_details":{"reason":"max_output_tokens"},
        "output":[item.clone()],
        "usage":{"input_tokens":3,"output_tokens":5}
    }))
    .unwrap();
    assert!(matches!(response.stop_reason, StopReason::MaxTokens));
    assert!(matches!(
        &response.message.content[..],
        [ContentBlock::ProviderContent { protocol: ProtocolFamily::OpenAiResponses, value }]
            if value == &item
    ));
    assert!(response.usage.usage.is_some());
}

#[test]
fn stream_publishes_one_call_only_after_consistent_completed_terminal_event() {
    let profile = profile();
    let mut decoder = computer_stream_decoder(&profile);
    let item = call_item("call_stream");
    let early = wire_api::decode_frame(
        &mut *decoder,
        json!({"type":"response.output_item.done","output_index":0,"item":item.clone()})
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    assert!(!early
        .iter()
        .any(|event| matches!(event, StreamEvent::Native { .. })));

    let terminal = json!({
        "type":"response.completed",
        "response":{"id":"resp_stream","status":"completed","output":[item.clone()]}
    });
    let events = wire_api::decode_frame(&mut *decoder, terminal.to_string().as_bytes()).unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::Native { value, .. }
                if OpenAiComputerCall::from_extension(value).is_ok()))
            .count(),
        1
    );
    let duplicate_terminal =
        wire_api::decode_frame(&mut *decoder, terminal.to_string().as_bytes()).unwrap();
    assert!(!duplicate_terminal
        .iter()
        .any(|event| matches!(event, StreamEvent::Native { .. })));
}

#[test]
fn completed_stream_must_reconcile_added_computer_identity_and_output() {
    let profile = profile();
    let mut added_item = call_item("call_added");
    added_item["status"] = json!("in_progress");
    added_item["actions"] = json!([]);
    let added = json!({"type":"response.output_item.added", "output_index":0,
        "item":added_item});

    for output in [
        None,
        Some(json!([])),
        Some(json!([call_item("call_replacement")])),
        Some(json!({"invalid":"output"})),
    ] {
        let mut decoder = computer_stream_decoder(&profile);
        assert!(push_stream_frame(&mut *decoder, &added)
            .iter()
            .all(Result::is_ok));
        let mut response = json!({"id":"resp_stream", "status":"completed"});
        if let Some(output) = output {
            response["output"] = output;
        }
        let events = push_stream_frame(
            &mut *decoder,
            &json!({"type":"response.completed", "response":response}),
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
        assert!(!events.iter().any(|event| matches!(
            event,
            Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
        )));
    }

    let mut decoder = computer_stream_decoder(&profile);
    assert!(push_stream_frame(&mut *decoder, &added)
        .iter()
        .all(Result::is_ok));
    let events = push_stream_frame(
        &mut *decoder,
        &json!({"type":"response.output_item.done", "output_index":0,
            "item":call_item("call_replacement")}),
    );
    assert!(events
        .iter()
        .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
    assert!(!events.iter().any(|event| matches!(
        event,
        Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
    )));

    let mut decoder = computer_stream_decoder(&profile);
    assert!(push_stream_frame(&mut *decoder, &added)
        .iter()
        .all(Result::is_ok));
    let events = push_stream_frame(
        &mut *decoder,
        &json!({"type":"response.completed", "response":{
            "id":"resp_stream", "status":"completed", "output":[call_item("call_added")]
        }}),
    );
    assert!(events
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
    assert!(events
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::End { .. }))));

    let mut decoder = computer_stream_decoder(&profile);
    assert!(push_stream_frame(&mut *decoder, &added)
        .iter()
        .all(Result::is_ok));
    let events = push_stream_frame(
        &mut *decoder,
        &json!({"type":"response.incomplete", "response":{
            "id":"resp_stream", "status":"incomplete", "output":[],
            "incomplete_details":{"reason":"max_output_tokens"}
        }}),
    );
    assert!(events.iter().all(Result::is_ok));
    assert!(!events
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
}

#[test]
fn premature_done_after_computer_item_is_stream_interruption() {
    let profile = profile();
    for stage in ["response.output_item.added", "response.output_item.done"] {
        let mut decoder = computer_stream_decoder(&profile);
        let item = if stage == "response.output_item.done" {
            call_item("call_premature_done")
        } else {
            json!({"type":"computer_call", "id":"cci_call_premature_done",
                "call_id":"call_premature_done", "status":"in_progress"})
        };
        let events = push_stream_frame(
            &mut *decoder,
            &json!({"type":stage, "output_index":0, "item":item}),
        );
        assert!(events.iter().all(Result::is_ok));
        let events = decoder.push_bytes(b"data: [DONE]\n\n");
        assert!(events
            .iter()
            .any(|event| matches!(event, Err(LlmError::StreamInterrupted { .. }))));
        assert!(!events.iter().any(|event| matches!(
            event,
            Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
        )));
    }
}

#[test]
fn completed_stream_call_requires_explicit_safety_checks_array() {
    let profile = profile();
    let mut item = call_item("call_stream_empty_checks");
    item["pending_safety_checks"] = json!([]);
    let mut decoder = computer_stream_decoder(&profile);
    let done = json!({"type":"response.output_item.done","output_index":0,"item":item.clone()});
    assert!(push_stream_frame(&mut *decoder, &done)
        .iter()
        .all(Result::is_ok));
    let terminal = json!({"type":"response.completed","response":{
        "id":"resp_stream", "status":"completed", "output":[item.clone()]
    }});
    let events = push_stream_frame(&mut *decoder, &terminal);
    assert!(events.iter().any(|event| matches!(event,
        Ok(StreamEvent::Native { value, .. })
            if value.data()["pending_safety_checks"] == json!([]))));

    item.as_object_mut()
        .unwrap()
        .remove("pending_safety_checks");
    for reject_at_done in [true, false] {
        let mut decoder = computer_stream_decoder(&profile);
        if reject_at_done {
            let events = push_stream_frame(
                &mut *decoder,
                &json!({"type":"response.output_item.done","output_index":0,"item":item}),
            );
            assert!(events
                .iter()
                .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
            assert!(!events.iter().any(|event| matches!(
                event,
                Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
            )));
        } else {
            let events = push_stream_frame(
                &mut *decoder,
                &json!({"type":"response.completed","response":{
                    "id":"resp_stream", "status":"completed", "output":[item]
                }}),
            );
            assert!(events
                .iter()
                .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
            assert!(!events.iter().any(|event| matches!(
                event,
                Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
            )));
        }
    }
}

#[test]
fn completed_stream_computer_item_must_match_terminal_actions_and_safety_checks() {
    let profile = profile();
    let observed = call_item("call_stream_consistency");
    for terminal_call in [
        {
            let mut call = observed.clone();
            call["actions"] = json!([{"type":"keypress","keys":["ENTER"]}]);
            call
        },
        {
            let mut call = observed.clone();
            call["pending_safety_checks"] = json!([]);
            call
        },
    ] {
        let mut decoder = computer_stream_decoder(&profile);
        let done = json!({"type":"response.output_item.done","output_index":0,
            "item":observed.clone()});
        assert!(push_stream_frame(&mut *decoder, &done)
            .iter()
            .all(Result::is_ok));
        let terminal = json!({"type":"response.completed","response":{
            "id":"resp_stream","status":"completed","output":[terminal_call]
        }});
        let events = push_stream_frame(&mut *decoder, &terminal);
        assert!(events
            .iter()
            .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
        assert!(!events.iter().any(|event| matches!(
            event,
            Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
        )));
    }
}

#[test]
fn stream_computer_call_requires_matching_created_and_terminal_response_ids() {
    let profile = profile();
    for (created_id, terminal_id) in [
        (Some("resp_a"), Some("resp_b")),
        (Some("resp_a"), None),
        (Some("resp_a"), Some("")),
        (Some(""), Some("resp_b")),
        (None, Some("resp_b")),
        (None, None),
    ] {
        let mut decoder =
            OpenAiResponsesCodec.stream_decoder(&context(&profile, RequestMode::Stream));
        if let Some(id) = created_id {
            let created = json!({"type":"response.created","response":{"id":id,"model":MODEL}});
            assert!(push_stream_frame(&mut *decoder, &created)
                .iter()
                .all(Result::is_ok));
        }
        let terminal = json!({"type":"response.completed","response":{
            "status":"completed","output":[call_item("call_bound")]
        }});
        let mut terminal = terminal;
        if let Some(id) = terminal_id {
            terminal["response"]["id"] = json!(id);
        }
        let events = push_stream_frame(&mut *decoder, &terminal);
        assert!(events
            .iter()
            .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
        assert!(!events.iter().any(|event| matches!(
            event,
            Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
        )));
    }
}

#[test]
fn completed_stream_computer_call_requires_item_id() {
    let profile = profile();
    let mut item = call_item("call_stream_no_item_id");
    item.as_object_mut().unwrap().remove("id");
    for reject_at_done in [true, false] {
        let mut decoder = computer_stream_decoder(&profile);
        let frame = if reject_at_done {
            json!({"type":"response.output_item.done", "output_index":0, "item":item})
        } else {
            json!({"type":"response.completed", "response":{
                "id":"resp_stream", "status":"completed", "output":[item]
            }})
        };
        let events = push_stream_frame(&mut *decoder, &frame);
        assert!(events
            .iter()
            .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
        assert!(!events.iter().any(|event| matches!(
            event,
            Ok(StreamEvent::Native { .. } | StreamEvent::End { .. })
        )));
    }
}

#[test]
fn mixed_client_actions_preserve_completed_computer_calls_in_stream_assembly() {
    use lingxi_llm_client::stream_assembly::StreamAccumulator;

    let profile = profile();
    for required_action in [
        json!({"type":"tool_search_call","id":"tsc_1","execution":"client","status":"completed"}),
        json!({"type":"mcp_approval_request","id":"mcpr_1","server_label":"remote"}),
    ] {
        let mut decoder = computer_stream_decoder(&profile);
        let response_body = json!({
            "id":"resp_stream",
            "status":"completed",
            "output":[call_item("call_mixed"), required_action]
        });
        let nonstream = decode(response_body.clone()).unwrap();
        assert_eq!(
            nonstream.stop_reason,
            StopReason::Other("requires_action".into())
        );
        assert_eq!(nonstream.message.content.len(), 2);
        let terminal = json!({
            "type":"response.completed",
            "response":response_body
        });
        let events =
            wire_api::decode_frame(&mut *decoder, terminal.to_string().as_bytes()).unwrap();
        let mut assembly = StreamAccumulator::new();
        for event in &events {
            assembly.observe(event);
        }
        assert_eq!(
            assembly.snapshot().response.stop_reason,
            StopReason::Other("requires_action".into())
        );
        let result = assembly.finish().unwrap();
        assert_eq!(result.response.message.content.len(), 2);
        assert_eq!(
            OpenAiComputerCall::from_content_block(&result.response.message.content[0])
                .unwrap()
                .call_id,
            "call_mixed"
        );
    }
}

#[test]
fn incomplete_mixed_response_never_materializes_a_computer_call() {
    use lingxi_llm_client::stream_assembly::StreamAccumulator;

    let profile = profile();
    let mut decoder = computer_stream_decoder(&profile);
    let terminal = json!({
        "type":"response.incomplete",
        "response":{
            "id":"resp_stream",
            "status":"incomplete",
            "output":[
                call_item("call_incomplete_mixed"),
                {"type":"tool_search_call","id":"tsc_1","execution":"client","status":"completed"}
            ]
        }
    });
    let events = wire_api::decode_frame(&mut *decoder, terminal.to_string().as_bytes()).unwrap();
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::Native { .. })));
    let mut assembly = StreamAccumulator::new();
    for event in &events {
        assembly.observe(event);
    }
    let result = assembly.finish().unwrap();
    assert!(result
        .response
        .message
        .content
        .iter()
        .all(|block| OpenAiComputerCall::from_content_block(block).is_err()));
}

#[test]
fn stream_rejects_status_mismatch_and_duplicate_ids_before_any_typed_call_event() {
    let profile = profile();
    for (event_type, status) in [
        ("response.completed", Some("incomplete")),
        ("response.completed", None),
        ("response.incomplete", Some("completed")),
    ] {
        let mut decoder = computer_stream_decoder(&profile);
        let mut response = json!({
            "id":"resp_stream",
            "output":[call_item("call_mismatch")],
            "usage":{"input_tokens":2,"output_tokens":3}
        });
        if let Some(status) = status {
            response["status"] = json!(status);
        }
        let events = push_stream_frame(
            &mut *decoder,
            &json!({"type":event_type,"response":response}),
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
        assert!(!events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
        let usage = decoder.usage_report().usage.unwrap();
        assert_eq!(usage.input_tokens, 2);
        assert_eq!(usage.output_tokens, 3);
    }

    let mut duplicate_decoder = computer_stream_decoder(&profile);
    let duplicate_results = push_stream_frame(
        &mut *duplicate_decoder,
        &json!({
            "type":"response.completed",
            "response":{"id":"resp_stream","status":"completed","output":[
                call_item("same_stream_call"),
                call_item("same_stream_call")
            ],"usage":{"input_tokens":2,"output_tokens":3}}
        }),
    );
    assert!(duplicate_results
        .iter()
        .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
    assert!(!duplicate_results
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
    assert_eq!(
        duplicate_decoder.usage_report().usage.unwrap().input_tokens,
        2
    );
}

#[test]
fn stream_validates_the_entire_call_batch_before_emitting_a_valid_prefix() {
    let profile = profile();
    let valid = call_item("call_valid_prefix");
    let mut unknown_action = call_item("call_invalid_suffix");
    unknown_action["actions"] = json!([{"type":"launch_app","name":"terminal"}]);

    for output in [
        vec![valid.clone(), unknown_action],
        vec![valid, call_item("call_duplicate_suffix")],
    ] {
        let mut output = output;
        if output[1]["call_id"] == "call_duplicate_suffix" {
            output[1]["call_id"] = output[0]["call_id"].clone();
        }
        let mut decoder = computer_stream_decoder(&profile);
        let results = push_stream_frame(
            &mut *decoder,
            &json!({
                "type":"response.completed",
                "response":{"id":"resp_stream","status":"completed","output":output,
                    "usage":{"input_tokens":2,"output_tokens":3}}
            }),
        );
        assert!(results.iter().any(Result::is_err));
        assert!(!results
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
        assert_eq!(decoder.usage_report().usage.unwrap().output_tokens, 3);
    }
}

#[test]
fn malformed_function_call_in_completed_batch_blocks_computer_execution_event() {
    let profile = profile();
    for malformed in [
        json!({"type":"function_call","name":"lookup","arguments":"{}"}),
        json!({"type":"function_call","call_id":"function_1","arguments":"{}"}),
        json!({"type":"function_call","call_id":"function_1","name":"lookup"}),
        json!({"type":"function_call","call_id":"function_1","name":"lookup","arguments":"{"}),
    ] {
        let response = json!({
            "id":"resp_stream",
            "status":"completed",
            "output":[call_item("computer_1"), malformed]
        });
        assert!(matches!(
            decode(response.clone()),
            Err(LlmError::InvalidRequest { .. })
        ));

        let mut decoder = computer_stream_decoder(&profile);
        let events = push_stream_frame(
            &mut *decoder,
            &json!({"type":"response.completed","response":response}),
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
        assert!(!events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
    }
}

#[test]
fn streamed_function_identity_must_match_terminal_batch_before_computer_call() {
    let profile = profile();
    for (added_id, added_name, streamed_args, terminal_id, terminal_name, terminal_args, valid) in [
        (
            "function_1",
            "lookup",
            "{}",
            "function_1",
            "lookup",
            "{}",
            true,
        ),
        (
            "computer_1",
            "lookup",
            "{}",
            "function_1",
            "lookup",
            "{}",
            false,
        ),
        (
            "function_1",
            "wrong",
            "{}",
            "function_1",
            "lookup",
            "{}",
            false,
        ),
        (
            "function_1",
            "lookup",
            "{\"x\":1}",
            "function_1",
            "lookup",
            "{\"x\":2}",
            false,
        ),
    ] {
        let mut decoder = computer_stream_decoder(&profile);
        for frame in [
            json!({"type":"response.output_item.added","output_index":1,
                "item":{"type":"function_call","call_id":added_id,"name":added_name}}),
            json!({"type":"response.function_call_arguments.delta","output_index":1,
                "delta":streamed_args}),
        ] {
            let events = push_stream_frame(&mut *decoder, &frame);
            assert!(events.iter().all(Result::is_ok));
        }
        let events = push_stream_frame(
            &mut *decoder,
            &json!({"type":"response.completed","response":{
                "id":"resp_stream",
                "status":"completed",
                "output":[call_item("computer_1"),
                    {"type":"function_call","call_id":terminal_id,"name":terminal_name,
                     "arguments":terminal_args}]
            }}),
        );
        if valid {
            assert!(events
                .iter()
                .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
        } else {
            assert!(events
                .iter()
                .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
            assert!(!events
                .iter()
                .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
        }
    }
}

#[test]
fn finalized_function_arguments_complete_mixed_computer_stream() {
    use lingxi_llm_client::stream_assembly::StreamAccumulator;

    let profile = profile();
    for (initial_arguments, final_arguments) in [
        ("", "{\"x\":1}"),
        ("{\"x\":1}", "{\"x\":1}"),
        ("{\"x\":1}", "{ \"x\": 1 }"),
    ] {
        let mut decoder = computer_stream_decoder(&profile);
        let mut assembly = StreamAccumulator::new();
        for frame in [
            json!({"type":"response.output_item.added","output_index":1,
                "item":{"type":"function_call","call_id":"function_1","name":"lookup",
                    "arguments":initial_arguments}}),
            json!({"type":"response.function_call_arguments.done","output_index":1,
                "arguments":final_arguments}),
            json!({"type":"response.output_item.done","output_index":1,
                "item":{"type":"function_call","call_id":"function_1","name":"lookup",
                    "arguments":final_arguments}}),
            json!({"type":"response.completed","response":{
                "id":"resp_stream",
                "status":"completed",
                "output":[call_item("computer_1"),
                    {"type":"function_call","call_id":"function_1","name":"lookup",
                        "arguments":final_arguments}]
            }}),
        ] {
            for event in
                wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap()
            {
                assembly.observe(&event);
            }
        }
        let response = assembly.finish().unwrap().response;
        assert!(response.message.content.iter().any(|block| matches!(block,
            ContentBlock::ToolUse { id, input, .. }
                if id.as_str() == "function_1" && input == &json!({"x":1}))));
        assert!(response
            .message
            .content
            .iter()
            .any(|block| OpenAiComputerCall::from_content_block(block).is_ok()));
    }
}

#[test]
fn finalized_function_arguments_reject_conflicting_stream_fragments() {
    let profile = profile();
    let mut decoder = computer_stream_decoder(&profile);
    for frame in [
        json!({"type":"response.output_item.added","output_index":1,
            "item":{"type":"function_call","call_id":"function_1","name":"lookup",
                "arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,
            "delta":"{\"x\":1}"}),
    ] {
        assert!(push_stream_frame(&mut *decoder, &frame)
            .iter()
            .all(Result::is_ok));
    }
    let done = push_stream_frame(
        &mut *decoder,
        &json!({"type":"response.function_call_arguments.done","output_index":1,
            "arguments":"{\"x\":2}"}),
    );
    assert!(done
        .iter()
        .any(|event| matches!(event, Err(LlmError::InvalidRequest { .. }))));
    let terminal = push_stream_frame(
        &mut *decoder,
        &json!({"type":"response.completed","response":{
            "id":"resp_stream",
            "status":"completed",
            "output":[call_item("computer_1"),
                {"type":"function_call","call_id":"function_1","name":"lookup",
                    "arguments":"{\"x\":2}"}]
        }}),
    );
    assert!(!terminal
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::Native { .. }))));
}

#[test]
fn returned_computer_outputs_accept_readback_fields_without_reusing_them_as_input() {
    let profile = profile();
    let call = OpenAiComputerCall::from_response_item(&call_item("call_readback")).unwrap();
    for fields in [json!({"status":"failed"}), json!({"created_by":"host_1"})] {
        let mut item = json!({
            "type":"computer_call_output",
            "call_id":"call_readback",
            "output":{"type":"computer_screenshot","image_url":"https://example.test/screen.png"}
        });
        item.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let output = OpenAiComputerCallOutput::from_response_item(&item).unwrap();
        let mut request = request();
        request.set_openai_computer_tool(Some(OpenAiComputerToolConfig::default()));
        request.continuation = Some(ContinuationRef {
            response_id: ResponseId::new("resp-previous"),
            provider_id: ProviderId::new("openai"),
            profile_name: "openai".into(),
            endpoint_fingerprint: "endpoint".into(),
            account_scope: "test-account".into(),
            request_model: MODEL.into(),
            workspace_id: None,
        });
        request.messages[0].content = vec![ContentBlock::Native {
            value: NativeExtension::from_typed(output.clone()).unwrap(),
        }];
        assert!(matches!(
            OpenAiResponsesCodec
                .validate_request(&request, &context(&profile, RequestMode::Complete)),
            Err(LlmError::InvalidRequest { .. })
        ));
        assert!(matches!(
            encode(&request, &profile),
            Err(LlmError::InvalidRequest { .. })
        ));
        assert!(output.into_content_block_for(&call).is_err());
    }
    let omitted_image = OpenAiComputerCallOutput::from_response_item(&json!({
        "type":"computer_call_output",
        "call_id":"call_readback",
        "status":"completed",
        "output":{"type":"computer_screenshot"}
    }))
    .unwrap();
    assert!(omitted_image.into_content_block_for(&call).is_err());
    let both_sources = OpenAiComputerCallOutput::from_response_item(&json!({
        "type":"computer_call_output",
        "call_id":"call_readback",
        "status":"completed",
        "output":{"type":"computer_screenshot","file_id":"file_1",
            "image_url":"https://example.test/screen.png"}
    }))
    .unwrap();
    assert!(both_sources.into_content_block_for(&call).is_err());
}

#[test]
fn request_validation_rejects_raw_computer_calls_before_encode() {
    let profile = profile();
    let mut request = request();
    request.messages[0]
        .content
        .push(ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: call_item("raw_call"),
        });
    assert!(matches!(
        OpenAiResponsesCodec.validate_request(&request, &context(&profile, RequestMode::Complete)),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn incomplete_terminal_stream_retains_raw_call_and_usage_without_native_execution_event() {
    let profile = profile();
    let mut decoder = computer_stream_decoder(&profile);
    let item = call_item("call_stream_incomplete");
    let events = wire_api::decode_frame(
        &mut *decoder,
        json!({
            "type":"response.incomplete",
            "response":{"id":"resp_stream","status":"incomplete","output":[item.clone()],"usage":{"input_tokens":2,"output_tokens":7}}
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::Native { .. })));
    assert!(events.iter().any(|event| matches!(event,
        StreamEvent::ProviderContent { protocol: ProtocolFamily::OpenAiResponses, value, .. }
            if value == &item)));
    assert!(matches!(events.last(), Some(StreamEvent::End { .. })));
    assert!(decoder.usage_report().usage.is_some());
}

#[test]
fn output_without_typed_tool_declaration_is_rejected_by_request_validator() {
    let profile = profile();
    let mut request = request();
    let output = serde_json::from_value::<OpenAiComputerCallOutput>(json!({
        "type":"computer_call_output",
        "call_id":"call_no_tool",
        "output":{"type":"computer_screenshot","image_url":"https://example.test/screen.png"}
    }))
    .unwrap()
    .into_content_block()
    .unwrap();
    request.messages[0].content = vec![output];
    request.continuation = Some(ContinuationRef {
        response_id: ResponseId::new("resp-previous"),
        provider_id: ProviderId::new("openai"),
        profile_name: "openai".into(),
        endpoint_fingerprint: "endpoint".into(),
        account_scope: "test-account".into(),
        request_model: MODEL.into(),
        workspace_id: None,
    });
    assert!(matches!(
        OpenAiResponsesCodec.validate_request(&request, &context(&profile, RequestMode::Complete)),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn tool_config_is_not_allowed_on_non_openai_responses_profile() {
    let mut foreign_profile = profile();
    foreign_profile.provider_id = ProviderId::new("xai");
    foreign_profile.profile_name = "xai-responses".into();
    let mut request = request();
    request.set_openai_computer_tool(Some(OpenAiComputerToolConfig::default()));
    assert!(matches!(
        OpenAiResponsesCodec
            .validate_request(&request, &context(&foreign_profile, RequestMode::Complete)),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn provider_owned_typed_call_can_be_constructed_for_replay_validation() {
    let call = OpenAiComputerCall::from_response_item(&call_item("call_helper")).unwrap();
    assert_eq!(call.actions, vec![ComputerAction::Screenshot]);
    let output = serde_json::from_value::<OpenAiComputerCallOutput>(json!({
        "type":"computer_call_output",
        "call_id":"call_helper",
        "output":{"type":"computer_screenshot","image_url":"https://example.test/screen.png"},
        "acknowledged_safety_checks":[{"id":"safety_1","code":"confirm","message":"Review this action"}]
    }))
    .unwrap();
    assert!(output.validate_for_call(&call).is_ok());
    let id_only_output = serde_json::from_value::<OpenAiComputerCallOutput>(json!({
        "type":"computer_call_output",
        "call_id":"call_helper",
        "output":{"type":"computer_screenshot","image_url":"https://example.test/screen.png"},
        "acknowledged_safety_checks":[{"id":"safety_1"}]
    }))
    .unwrap();
    assert!(id_only_output.into_content_block_for(&call).is_ok());
    assert_eq!(
        call.pending_safety_checks,
        vec![ComputerSafetyCheck {
            id: "safety_1".into(),
            code: Some("confirm".into()),
            message: Some("Review this action".into()),
        }]
    );
}
