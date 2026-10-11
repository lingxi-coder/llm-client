use lingxi_llm_client::codecs::{
    anthropic::AnthropicMessagesCodec, CodecContext, RequestMode, WireCodec,
};
use lingxi_llm_client::protocol::{
    ChatResponse, ContentBlock, NativeExtension, ProtocolFamily, ProviderProfile, StreamEvent,
};
use lingxi_llm_client::providers::anthropic::fallback_response::{
    self, FallbackControl, FallbackHop, ServerFallbackEvent,
};
use lingxi_llm_client::stream_assembly::StreamAccumulator;
use lingxi_llm_client::transport::HttpResponse;
use serde_json::{json, Value};

const FIXTURE: &str = include_str!("fixtures/server_fallback_projection_2_1_288.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

fn decoder() -> (AnthropicMessagesCodec, CodecContext) {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "custom",
        "profile_name": "fixture",
        "base_url": "https://fixture.invalid",
        "protocol": "anthropic_messages",
        "auth": "none",
        "models": [{
            "display_model": "primary",
            "request_model": "primary",
            "billing_model": "primary"
        }]
    }))
    .unwrap();
    let context = CodecContext::for_model(&profile, &profile.models[0], RequestMode::Complete);
    (AnthropicMessagesCodec, context)
}

fn decode_body(
    codec: &AnthropicMessagesCodec,
    context: &CodecContext,
    body: &Value,
) -> ChatResponse {
    codec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: serde_json::to_vec(body).unwrap().into(),
            },
            context,
        )
        .unwrap()
}

fn completed_blocks(input: &Value) -> Vec<(usize, ContentBlock)> {
    input["completed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let index = usize::try_from(entry[0].as_u64().unwrap()).unwrap();
            (
                index,
                ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::AnthropicMessages,
                    value: entry[1].clone(),
                },
            )
        })
        .collect()
}

fn parse_hop(value: &Value) -> Option<FallbackHop> {
    (!value.is_null()).then(|| serde_json::from_value(value.clone()).unwrap())
}

fn projected_stream_event(input: &Value) -> Option<ServerFallbackEvent> {
    let completed = completed_blocks(input);
    let hop = parse_hop(&input["hop"]);
    let request_id = input["requestId"].as_str().map(str::to_owned);
    if !completed.is_empty() {
        if let Some(hop) = hop.as_ref() {
            return Some(fallback_response::stream_event(hop, &completed, request_id));
        }
    }

    let iterations = input
        .get("usage")
        .filter(|usage| usage["iterations"].is_array());
    let iterations = iterations.map(fallback_response::usage_iterations);
    let stop_reason = if input["interrupted"] == true {
        None
    } else {
        input["stopReason"].as_str().map(str::to_owned)
    };
    fallback_response::terminal_event(
        "primary",
        hop.as_ref(),
        iterations.as_ref(),
        request_id,
        stop_reason,
    )
}

#[test]
fn stream_projection_helpers_match_native_2_1_288_cases() {
    for (case_index, case) in fixture()["streamCases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let input = &case["input"];
        let expected_events = case["events"].as_array().unwrap();
        if input["armed"] != true {
            assert!(expected_events.is_empty(), "unarmed case {case_index}");
            continue;
        }

        let event = projected_stream_event(input);
        let actual_events = event.iter().collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(actual_events).unwrap(),
            Value::Array(expected_events.clone()),
            "stream event case {case_index}"
        );

        let discarded = event
            .as_ref()
            .map(|event| event.discarded_blocks.as_slice())
            .unwrap_or_default();
        let remaining = input["completed"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|block| {
                let index = usize::try_from(block[0].as_u64().unwrap()).unwrap();
                (!discarded.contains(&index)).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(remaining).unwrap(),
            case["remaining"],
            "remaining completed blocks in case {case_index}"
        );
    }
}

#[test]
fn nonstream_projection_and_unarmed_decode_match_native_2_1_288_cases() {
    let (codec, context) = decoder();
    for (case_index, case) in fixture()["nonstreamCases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let input = &case["input"];
        let body = &input["body"];
        let mut response = decode_body(&codec, &context, body);

        let event = if input["armed"] == true {
            let response_id = response
                .response_id
                .as_ref()
                .map(|response_id| response_id.as_str().to_owned());
            fallback_response::project_nonstream(
                &mut response,
                body["model"].as_str().unwrap_or_default(),
                response_id,
            )
        } else {
            None
        };
        assert_eq!(
            serde_json::to_value(&event).unwrap(),
            case["event"],
            "nonstream event case {case_index}"
        );

        let mut expected_body = body.clone();
        expected_body["content"] = case["content"].clone();
        let expected = decode_body(&codec, &context, &expected_body);
        assert_eq!(
            response.message.content, expected.message.content,
            "nonstream content case {case_index}"
        );
        if let Some(event) = event {
            assert_eq!(response.model, event.to_model, "model case {case_index}");
        } else {
            assert_eq!(
                response.model,
                body["model"].as_str().unwrap_or_default(),
                "unprojected model case {case_index}"
            );
        }
    }
}

#[test]
fn stream_accumulator_preserves_completion_order_and_requires_explicit_admission() {
    let control = NativeExtension::from_typed(FallbackControl::Start {
        start: serde_json::from_value(json!({
            "index": 4,
            "fromModel": "primary",
            "model": "served",
            "reason": "refusal",
            "category": "cyber"
        }))
        .unwrap(),
    })
    .unwrap();
    let mut accumulator = StreamAccumulator::new();
    accumulator.observe(&StreamEvent::Start {
        model: "primary".into(),
        response_id: Some("resp-1".into()),
    });
    accumulator.observe(&StreamEvent::NativeControl {
        protocol: ProtocolFamily::AnthropicMessages,
        control,
    });
    assert!(accumulator.completed_blocks().is_empty());
    assert_eq!(accumulator.snapshot().response.model, "primary");

    for (index, text) in [(9, "later"), (2, "earlier")] {
        accumulator.observe(&StreamEvent::TextDelta {
            block: index,
            text: text.into(),
        });
        accumulator.observe(&StreamEvent::BlockEnd { block: index });
    }
    accumulator.observe(&StreamEvent::ProviderContent {
        block: 5,
        protocol: ProtocolFamily::AnthropicMessages,
        value: json!({"type":"thinking","thinking":"private"}),
    });
    accumulator.observe(&StreamEvent::BlockEnd { block: 5 });
    assert_eq!(
        accumulator
            .completed_blocks()
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>(),
        vec![9, 2, 5]
    );

    let hop = FallbackHop {
        from_model: "primary".into(),
        model: "served".into(),
        reason: "refusal".into(),
        category: Some("cyber".into()),
    };
    let event = fallback_response::stream_event(&hop, &accumulator.completed_blocks(), None);
    accumulator.apply_server_fallback(&event);
    let after = accumulator.completed_blocks();
    assert_eq!(after.len(), 2);
    assert_eq!(
        after.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        vec![9, 2]
    );
    assert_eq!(accumulator.snapshot().response.model, "served");

    accumulator.set_response_model("second-target");
    assert_eq!(accumulator.snapshot().response.model, "second-target");
}

#[test]
fn nonstream_projection_keeps_response_id_and_terminal_reason() {
    let (codec, context) = decoder();
    let body = json!({
        "id": "msg-1",
        "model": "primary",
        "content": [{
            "type": "fallback",
            "from": {"model": "primary"},
            "to": {"model": "served"},
            "trigger": {"type": "refusal", "category": "cyber"}
        }],
        "stop_reason": "refusal",
        "usage": {}
    });
    let mut response = decode_body(&codec, &context, &body);
    let response_id = response
        .response_id
        .as_ref()
        .map(|response_id| response_id.as_str().to_owned());
    let event =
        fallback_response::project_nonstream(&mut response, "primary", response_id).unwrap();
    assert_eq!(event.request_id.as_deref(), Some("msg-1"));
    assert_eq!(event.final_stop_reason.as_deref(), Some("refusal"));
    assert!(response.message.content.is_empty());
}

#[test]
fn admitted_fallback_response_id_is_reflected_in_later_snapshots() {
    let mut accumulator = StreamAccumulator::new();
    accumulator.observe(&StreamEvent::Start {
        model: "primary".into(),
        response_id: Some("resp-original".into()),
    });

    // The host calls this only for an admitted fallback start with an id.
    accumulator.set_response_id("resp-fallback");
    accumulator.set_response_model("served");

    let response = accumulator.snapshot().response;
    assert_eq!(response.model, "served");
    assert_eq!(
        response.response_id.as_ref().map(|id| id.as_str()),
        Some("resp-fallback")
    );
}
