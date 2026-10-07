use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, providers::google::computer::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
fn profile(name: &str) -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"google","profile_name":name,"protocol":"gemini_interactions","base_url":format!("https://{name}.test/v1beta"),"auth":"api_key","connection":{"group":"gemini","connection_id":name,"failover":{"network":true}},"models":[{"request_model":"gemini-3.8-flash","display_model":"m","billing_model":"gemini-3.8-flash"}]})).unwrap()
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"computer","description":"control","input_schema":{"type":"object"}}]})).unwrap()
}
fn context(req: &ChatRequest, mode: RequestMode) -> CodecContext {
    CodecContext::new(&profile("one"), "gemini-3.8-flash", mode)
        .with_account_scope(Some("account-a"))
        .with_native_options(&req.native_options)
}
fn frame() -> ComputerFrame {
    ComputerFrame {
        width: 2000,
        height: 1000,
        geometry_version: "screen-1".into(),
    }
}

#[test]
fn malformed_final_images_cannot_form_gemini_computer_receipts() {
    let call = lingxi_llm_client::providers::google::computer::decode(
        &[ContentBlock::Native {
            value: NativeExtension::new(
                CALL_FORMAT,
                json!({"type":"function_call","id":"shot","name":"take_screenshot","arguments":{}}),
            )
            .unwrap(),
        }],
        None,
        &frame(),
    )
    .unwrap()
    .remove(0);
    for image in [
        json!({"type":"image","source":{"type":"base64","media_type":"text/plain","data":""}}),
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"not-base64"}}),
        json!({"type":"image","data":"","mime_type":"image/png"}),
        json!({"type":"image","data":"not-base64","mime_type":"image/png"}),
        json!({"type":"image","uri":""}),
        json!({"type":"image","uri":"https://example.test/shot.png","mime_type":"text/plain"}),
        json!({"type":"image","source":{"type":"url","url":"https://user:secret@example.test/a.png"}}),
    ] {
        let input = ComputerReceiptInput {
            results: vec![NativeComputerResult {
                operation_index: 0,
                status: NativeExecutionStatus::Succeeded,
                content: "shot".into(),
                blocks: Some(vec![image.clone()]),
            }],
            acknowledged_safety_checks: vec![],
        };
        assert!(
            encode_receipt(&call, &input).is_err(),
            "accepted invalid screenshot {image}"
        );
    }
    let input = ComputerReceiptInput {
        results: vec![NativeComputerResult {
            operation_index: 0,
            status: NativeExecutionStatus::Succeeded,
            content: "shot".into(),
            blocks: Some(vec![
                json!({"type":"image","uri":"https://example.test/shot.png","mime_type":"image/png"}),
            ]),
        }],
        acknowledged_safety_checks: vec![],
    };
    assert!(
        encode_receipt(&call, &input).is_ok(),
        "native URI images retain optional MIME"
    );
}
fn capabilities() -> ComputerCapabilities {
    ComputerCapabilities {
        operations: vec![
            ComputerOperationKind::Click,
            ComputerOperationKind::Move,
            ComputerOperationKind::Drag,
            ComputerOperationKind::Scroll,
            ComputerOperationKind::Key,
            ComputerOperationKind::KeyDown,
            ComputerOperationKind::KeyUp,
            ComputerOperationKind::Type,
            ComputerOperationKind::Wait,
            ComputerOperationKind::Screenshot,
            ComputerOperationKind::MouseDown,
            ComputerOperationKind::MouseUp,
        ],
    }
}
fn desktop() -> ChatRequest {
    let mut req = request();
    req.tools.clear();
    declare(&mut req, &capabilities(), &frame()).unwrap();
    req
}
fn full(step: Value) -> Value {
    json!({"id":"int_one","model":"gemini-3.8-flash","status":"requires_action","steps":[step],"usage":{"total_input_tokens":10,"total_cached_tokens":3,"total_output_tokens":2,"total_thought_tokens":4,"total_tool_use_tokens":0,"total_tokens":16}})
}
fn decode(body: &Value, req: &ChatRequest) -> ChatResponse {
    GeminiInteractionsCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(body).unwrap().into(),
            },
            &context(req, RequestMode::Complete),
        )
        .unwrap()
}
fn bytes(event: Value) -> Vec<u8> {
    format!("data: {event}\n\n").into_bytes()
}
#[test]
fn functions_and_desktop_share_endpoint_and_preserve_function_ids() {
    let req = request();
    let wire = GeminiInteractionsCodec
        .encode_request(
            EncodeRequest::new(&req),
            &context(&req, RequestMode::Complete),
        )
        .unwrap();
    assert_eq!(wire.url, "https://one.test/v1beta/interactions");
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    assert_eq!(body["tools"][0]["type"], "function");
    let native = desktop();
    let wire = GeminiInteractionsCodec
        .encode_request(
            EncodeRequest::new(&native),
            &context(&native, RequestMode::Complete),
        )
        .unwrap();
    assert_eq!(wire.url, "https://one.test/v1beta/interactions");
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    assert_eq!(body["tools"][0]["environment"], "desktop");
    let response = decode(
        &full(
            json!({"type":"function_call","id":"call_function","name":"computer","arguments":{"action":"screenshot"}}),
        ),
        &req,
    );
    assert!(
        matches!(&response.message.content[0],ContentBlock::ToolUse{id,provider_id:Some(provider_id),..}if id.as_str()=="call_function"&&provider_id=="call_function")
    );
    assert_eq!(response.usage.state, UsageState::Complete);
    assert_eq!(response.usage.usage.unwrap().input_tokens, 7);
    assert_eq!(response.usage.usage.unwrap().cache_read_tokens, 3);
    assert_eq!(response.usage.usage.unwrap().output_tokens, 6);
    assert_eq!(response.usage.usage.unwrap().reasoning_tokens, 4);
    let mut custom = req.clone();
    custom.tools[0].name = "click".into();
    assert!(matches!(
        &decode(
            &full(
                json!({"type":"function_call","id":"regular_click","name":"click","arguments":{}})
            ),
            &custom
        )
        .message
        .content[0],
        ContentBlock::ToolUse { .. }
    ));
}
#[test]
fn desktop_coordinates_defaults_and_ordered_observation() {
    let req = desktop();
    for (name, args) in [
        ("click", json!({"x":999,"y":500})),
        ("wait", json!({})),
        ("scroll", json!({"x":0,"y":0,"direction":"down"})),
    ] {
        let response = decode(
            &full(json!({"type":"function_call","id":"native","name":name,"arguments":args})),
            &req,
        );
        let calls = decode_computer_calls(
            NativeComputerProvider::Gemini,
            &response.message.content,
            None,
            &frame(),
        )
        .unwrap();
        let call = &calls[0];
        assert_eq!(call.context.action_count, 2);
        assert!(matches!(
            call.operations.last(),
            Some(ComputerOperation::Screenshot)
        ));
        match &call.operations[0] {
            ComputerOperation::Click {
                target: ComputerTarget::Position { point },
                ..
            } => assert_eq!(
                *point,
                ComputerPoint {
                    x: 1998.0,
                    y: 500.0
                }
            ),
            ComputerOperation::Wait { duration_seconds } => assert_eq!(*duration_seconds, 1.0),
            ComputerOperation::Scroll { delta_y, .. } => assert_eq!(*delta_y, 300.0),
            _ => panic!(),
        }
    }
    let response = decode(
        &full(
            json!({"type":"function_call","id":"invalid","name":"click","arguments":{"x":1000,"y":0}}),
        ),
        &req,
    );
    assert!(decode_computer_calls(
        NativeComputerProvider::Gemini,
        &response.message.content,
        None,
        &frame()
    )
    .is_err());
}
#[test]
fn native_stream_never_delivers_calls_before_terminal_and_fragmented_args_agree() {
    let req = desktop();
    let ctx = context(&req, RequestMode::Stream);
    let prefix=[json!({"event_type":"interaction.created","interaction":{"id":"int_one","model":"gemini-3.8-flash"}}),json!({"event_type":"step.start","index":0,"step":{"type":"function_call","id":"native","name":"click","arguments":{}}}),json!({"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{\"x\":999,"}}),json!({"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"\"y\":500}"}}),json!({"event_type":"step.stop","index":0})].into_iter().flat_map(bytes).collect::<Vec<_>>();
    for chunk_size in [1, 2, 7, 4096] {
        let mut decoder = GeminiInteractionsCodec.stream_decoder(&ctx);
        let mut events = vec![];
        for chunk in prefix.chunks(chunk_size) {
            events.extend(decoder.push_bytes(chunk));
        }
        assert!(events.iter().all(|event| !matches!(
            event,
            Ok(StreamEvent::Native { .. } | StreamEvent::ToolCallDelta { .. })
        )));
        let terminal = bytes(
            json!({"event_type":"interaction.completed","interaction":{"id":"int_one","status":"requires_action","usage":{"total_input_tokens":10,"total_output_tokens":2,"total_thought_tokens":4,"total_cached_tokens":3,"total_tokens":16}}}),
        );
        for chunk in terminal.chunks(chunk_size) {
            events.extend(decoder.push_bytes(chunk));
        }
        events.extend(decoder.finish());
        let events = events.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::Native { .. }))
                .count(),
            1
        );
        assert_eq!(decoder.usage_report().state, UsageState::Complete);
    }
    let mut decoder = GeminiInteractionsCodec.stream_decoder(&ctx);
    decoder.push_bytes(&prefix);
    assert!(decoder.finish().iter().any(Result::is_err));
}

#[test]
fn terminal_steps_must_agree_with_streamed_calls_before_any_call_escapes() {
    let req = desktop();
    let ctx = context(&req, RequestMode::Stream);
    let call =
        json!({"type":"function_call","id":"native","name":"click","arguments":{"x":999,"y":500}});
    let other = json!({"type":"function_call","id":"other","name":"type_text","arguments":{"text":"hello"}});
    let mut safety = call.clone();
    safety["arguments"]["safety_decision"] =
        json!({"decision":"require_confirmation","explanation":"submit"});
    let mut changed = call.clone();
    changed["arguments"]["x"] = json!(0);
    for terminal in [
        json!([]),
        json!([other.clone(), call.clone()]),
        json!([safety, other.clone()]),
        json!([changed, other.clone()]),
        json!(null),
    ] {
        let mut decoder = GeminiInteractionsCodec.stream_decoder(&ctx);
        let mut events = Vec::new();
        for event in [
            json!({"event_type":"interaction.created","interaction":{"id":"int_one","model":"gemini-3.8-flash"}}),
            json!({"event_type":"step.start","index":0,"step":call}),
            json!({"event_type":"step.stop","index":0}),
            json!({"event_type":"step.start","index":1,"step":other}),
            json!({"event_type":"step.stop","index":1}),
            json!({"event_type":"interaction.completed","interaction":{"id":"int_one","status":"requires_action","steps":terminal}}),
        ] {
            events.extend(decoder.push_bytes(&bytes(event)));
        }
        assert!(events.iter().any(Result::is_err));
        assert!(events.iter().all(|event| !matches!(
            event,
            Ok(StreamEvent::Native { .. } | StreamEvent::ToolCallDelta { .. })
        )));
    }
    let mut decoder = GeminiInteractionsCodec.stream_decoder(&ctx);
    let mut events = Vec::new();
    for event in [
        json!({"event_type":"interaction.created","interaction":{"id":"int_one"}}),
        json!({"event_type":"step.start","index":0,"step":call}),
        json!({"event_type":"step.stop","index":0}),
        json!({"event_type":"interaction.completed","interaction":{"id":"int_one","status":"requires_action","steps":[call]}}),
    ] {
        events.extend(decoder.push_bytes(&bytes(event)));
    }
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Ok(StreamEvent::Native { .. })))
            .count(),
        1
    );
}

#[test]
fn ordinary_screenshot_results_translate_base64_and_url_content() {
    let mut req = request();
    req.messages.extend([
        serde_json::from_value(json!({"role":"assistant","content":[{"type":"tool_use","id":"shot","name":"computer","input":{"action":"screenshot"}}]})).unwrap(),
        serde_json::from_value(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"shot","content":"shot","blocks":[{"type":"text","text":"shot"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}},{"type":"image","source":{"type":"url","url":"https://one.test/shot.png"}}]}]})).unwrap(),
    ]);
    let wire = GeminiInteractionsCodec
        .encode_request(
            EncodeRequest::new(&req),
            &context(&req, RequestMode::Complete),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    let result = &body["input"][2]["result"];
    assert_eq!(
        result[1],
        json!({"type":"image","mime_type":"image/png","data":"aGVsbG8="})
    );
    assert_eq!(
        result[2],
        json!({"type":"image","uri":"https://one.test/shot.png"})
    );
}
#[test]
fn safety_receipt_requires_final_image_and_exact_confirmation() {
    let req = desktop();
    let check = json!({"decision":"require_confirmation","explanation":"submit form"});
    let response = decode(
        &full(
            json!({"type":"function_call","id":"native","name":"click","arguments":{"x":0,"y":0,"safety_decision":check}}),
        ),
        &req,
    );
    let calls = decode_computer_calls(
        NativeComputerProvider::Gemini,
        &response.message.content,
        None,
        &frame(),
    )
    .unwrap();
    let call = &calls[0];
    let mut input = ComputerReceiptInput {
        results: (0..2)
            .map(|index| NativeComputerResult {
                operation_index: index,
                status: NativeExecutionStatus::Succeeded,
                content: "done".into(),
                blocks: None,
            })
            .collect(),
        acknowledged_safety_checks: vec![check],
    };
    assert!(encode_receipt(call, &input).is_err());
    input.results[1].blocks = Some(vec![
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}),
    ]);
    let receipt = encode_receipt(call, &input).unwrap();
    let ContentBlock::Native { value } = receipt else {
        panic!()
    };
    assert_eq!(value.data()["call_id"], "native");
    assert_eq!(value.data()["name"], "click");
    assert_eq!(
        value.data()["result"].as_array().unwrap().last().unwrap()["mime_type"],
        "image/png"
    );
    input.acknowledged_safety_checks.clear();
    assert!(encode_receipt(call, &input).is_err());
    input.results[0].status = NativeExecutionStatus::OutcomeUnknown;
    input.results[1].status = NativeExecutionStatus::Skipped;
    assert!(encode_receipt(call, &input).is_err());
}
#[test]
fn an_earlier_image_cannot_replace_the_terminal_screenshot_after_hooks() {
    let response = decode(
        &full(
            json!({"type":"function_call","id":"native","name":"click","arguments":{"x":1,"y":1}}),
        ),
        &desktop(),
    );
    let call = decode_computer_calls(
        NativeComputerProvider::Gemini,
        &response.message.content,
        None,
        &frame(),
    )
    .unwrap()
    .remove(0);
    assert!(matches!(
        call.operations.last(),
        Some(ComputerOperation::Screenshot)
    ));
    let mut input = ComputerReceiptInput {
        results: (0..2)
            .map(|operation_index| NativeComputerResult {
                operation_index,
                status: NativeExecutionStatus::Succeeded,
                content: "done".into(),
                blocks: None,
            })
            .collect(),
        acknowledged_safety_checks: vec![],
    };
    let earlier = json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"ZWFybGllcg=="}});
    input.results[0].blocks = Some(vec![earlier]);
    assert!(
        encode_receipt(&call, &input).is_err(),
        "an image attached to the earlier click must not satisfy the terminal Screenshot"
    );
    input.results[1].blocks = Some(vec![
        json!({"type":"text","text":"final screenshot removed by hook"}),
    ]);
    assert!(encode_receipt(&call, &input).is_err());
    input.results[1].blocks = Some(vec![
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"dGVybWluYWw="}}),
    ]);
    assert!(encode_receipt(&call, &input).is_ok());
}

#[derive(Default)]
struct Mock {
    sent: Mutex<Vec<HttpRequest>>,
    fail: bool,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        if self.fail {
            return Err(LlmError::Transport {
                message: "lost after send".into(),
            });
        }
        let body=serde_json::to_vec(&json!({"id":"int_one","status":"completed","model":"gemini-3.8-flash","steps":[],"usage":{"total_input_tokens":1,"total_output_tokens":1,"total_tokens":2}})).unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(body.into()) }).boxed(),
        })
    }
}
fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("fixture-key".into())),
        account_scope: Some("account-a".into()),
        ..Default::default()
    }
}
#[tokio::test]
async fn prepared_unified_call_authenticates_and_continuation_is_scoped() {
    let http = Arc::new(Mock::default());
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile("one"), profile("two")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let response = client
        .prepare_on("one", &request(), &options(), RequestMode::Complete)
        .await
        .unwrap()
        .dispatch_once()
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .decode()
        .unwrap();
    let reference = response.continuation.unwrap();
    assert_eq!(reference.protocol, ProtocolFamily::GeminiInteractions);
    assert!(http.sent.lock().unwrap()[0]
        .headers
        .iter()
        .any(|(key, value)| key == "x-goog-api-key" && value == "fixture-key"));
    let mut req = request();
    req.continuation = Some(reference.clone());
    let next = client
        .prepare_on("one", &req, &options(), RequestMode::Complete)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&next.request().body).unwrap();
    assert_eq!(body["previous_interaction_id"], "int_one");
    for wrong in ["two"] {
        assert!(client
            .prepare_on(wrong, &req, &options(), RequestMode::Complete)
            .await
            .is_err());
    }
    let mut changed = options();
    changed.account_scope = Some("account-b".into());
    assert!(client
        .prepare_on("one", &req, &changed, RequestMode::Complete)
        .await
        .is_err());
    req.continuation.as_mut().unwrap().protocol = ProtocolFamily::OpenAiResponses;
    assert!(client
        .prepare_on("one", &req, &options(), RequestMode::Complete)
        .await
        .is_err());
    assert_eq!(http.sent.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn uncertain_interaction_submission_never_fails_over() {
    let http = Arc::new(Mock {
        fail: true,
        ..Default::default()
    });
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile("one"), profile("two")])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(client
        .chat()
        .complete_in("gemini", &request(), &options())
        .await
        .is_err());
    assert_eq!(http.sent.lock().unwrap().len(), 1);
}

#[test]
fn native_receipt_then_function_declaration_uses_previous_interaction_and_old_binding() {
    let req = desktop();
    let response = decode(
        &full(
            json!({"type":"function_call","id":"native_scroll","name":"scroll","arguments":{"x":10,"y":20,"direction":"up"}}),
        ),
        &req,
    );
    let calls = decode_computer_calls(
        NativeComputerProvider::Gemini,
        &response.message.content,
        None,
        &frame(),
    )
    .unwrap();
    let mut call = calls[0].clone();
    let input = ComputerReceiptInput {
        results: vec![
            NativeComputerResult {
                operation_index: 0,
                status: NativeExecutionStatus::Succeeded,
                content: "scrolled".into(),
                blocks: None,
            },
            NativeComputerResult {
                operation_index: 1,
                status: NativeExecutionStatus::Succeeded,
                content: "observed".into(),
                blocks: Some(vec![
                    json!({"type":"image","data":"aGVsbG8=","mime_type":"image/png"}),
                ]),
            },
        ],
        acknowledged_safety_checks: vec![],
    };
    let mut follow = request();
    follow.messages = vec![ConversationMessage {
        role: MessageRole::User,
        content: vec![],
        native_options: vec![],
    }];
    follow.continuation = Some(ContinuationRef {
        response_id: "int_one".into(),
        protocol: ProtocolFamily::GeminiInteractions,
        provider_id: "google".into(),
        profile_name: "one".into(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            "https://one.test/v1beta",
        ),
        account_scope: "account-a".into(),
        request_model: "gemini-3.8-flash".into(),
        workspace_id: None,
    });
    call.context.continuation = follow.continuation.clone();
    follow.messages[0]
        .content
        .push(encode_receipt(&call, &input).unwrap());
    let wire = GeminiInteractionsCodec
        .encode_request(
            EncodeRequest::new(&follow),
            &context(&follow, RequestMode::Complete),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    assert_eq!(body["previous_interaction_id"], "int_one");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["input"][0]["call_id"], "native_scroll");
    assert_eq!(body["input"][0]["name"], "scroll");
    assert!(body["input"][0].get("_sdk_continuation").is_none());
    follow.continuation.as_mut().unwrap().response_id = "different_interaction".into();
    assert!(GeminiInteractionsCodec
        .encode_request(
            EncodeRequest::new(&follow),
            &context(&follow, RequestMode::Complete)
        )
        .is_err());
}

#[test]
fn native_receipts_survive_same_protocol_history_projection_and_reject_foreign_routes() {
    use lingxi_llm_client::replay::{ReplayContext, ReplayPolicy};
    let receipt=ContentBlock::Native { value:NativeExtension::new(RESULT_FORMAT,json!({"type":"function_result","name":"click","call_id":"native","result":[{"type":"text","text":"denied"}]})).unwrap() };
    let same = ReplayContext::for_message(
        std::slice::from_ref(&receipt),
        ProtocolFamily::GeminiInteractions,
    );
    assert_eq!(
        same.normalize(
            &receipt,
            Some(ProtocolFamily::GeminiInteractions),
            ReplayPolicy::Reject
        )
        .unwrap(),
        Some(receipt.clone())
    );
    let foreign = ReplayContext::for_message(
        std::slice::from_ref(&receipt),
        ProtocolFamily::GeminiGenerateContent,
    );
    assert!(foreign
        .normalize(
            &receipt,
            Some(ProtocolFamily::GeminiInteractions),
            ReplayPolicy::Reject
        )
        .is_err());
}
