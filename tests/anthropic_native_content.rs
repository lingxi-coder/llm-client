use lingxi_llm_client::protocol::{
    ChatRequest, ContentBlock, ConversationMessage, LlmError, MessageRole, ProtocolFamily,
    ProviderProfile, StreamEvent, ToolChoice,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, CodecContext, EncodeRequest, HttpResponse, RequestMode, WireCodec,
};
use serde_json::{json, Value};

#[path = "support/wire_api.rs"]
mod wire_api;

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "anthropic",
        "profile_name": "anthropic",
        "base_url": "https://api.anthropic.test",
        "protocol": "anthropic_messages",
        "auth": "none",
        "models": [{"display_model":"claude-sonnet-4-5", "request_model":"claude-sonnet-4-5", "billing_model":"claude-sonnet-4-5"}],
        "extra": {}
    }))
    .unwrap()
}

fn request(content: Vec<ContentBlock>) -> ChatRequest {
    ChatRequest {
        prompt_cache: Default::default(),
        output_format: Default::default(),
        service_tier: None,
        model: "claude-sonnet-4-5".into(),
        anthropic_client_toolsets: Vec::new(),
        hosted_tools: vec![],
        continuation: None,
        system: vec![],
        messages: vec![ConversationMessage {
            anthropic: None,
            role: MessageRole::Assistant,
            content,
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: Some(256),
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}

fn encode(content: Vec<ContentBlock>) -> Result<Value, LlmError> {
    let profile = profile();
    let context = CodecContext::new(&profile, "claude-sonnet-4-5", RequestMode::Complete);
    let request = request(content);
    let wire = AnthropicMessagesCodec.encode_request(EncodeRequest::new(&request), &context)?;
    Ok(serde_json::from_slice(&wire.body).unwrap())
}

fn decode_stream(frames: &[Value]) -> Result<Vec<StreamEvent>, LlmError> {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    let mut events = Vec::new();
    for frame in frames {
        events.extend(wire_api::decode_frame(
            &mut *decoder,
            frame.to_string().as_bytes(),
        )?);
    }
    events.extend(wire_api::finish(&mut *decoder)?);
    Ok(events)
}

fn provider_events(events: &[StreamEvent]) -> Vec<&Value> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ProviderEvent {
                protocol: ProtocolFamily::AnthropicMessages,
                payload,
            } => Some(payload),
            _ => None,
        })
        .collect()
}

#[test]
fn native_server_results_and_future_blocks_decode_and_replay_unchanged() {
    let native = vec![
        json!({
            "type":"server_tool_use",
            "id":"srvtoolu_1",
            "name":"web_fetch",
            "input":{"url":"https://example.test"},
            "future_field":{"preserve":[1,2,3]}
        }),
        json!({
            "type":"web_fetch_tool_result",
            "tool_use_id":"srvtoolu_1",
            "content":[{"type":"text","text":"Fetched page"}],
            "is_error":false
        }),
        json!({
            "type":"provider_added_result_next_year",
            "opaque":{"version":3,"value":"keep me"}
        }),
        json!({
            "type":"text",
            "text":"A cited answer",
            "citations":[{"type":"char_location","start_char_index":0,"end_char_index":6,"document_index":0}]
        }),
    ];
    let body = json!({
        "model":"claude-sonnet-4-5",
        "stop_reason":"end_turn",
        "content":native
    });
    let response = AnthropicMessagesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            },
            &wire_api::decode_context(),
        )
        .unwrap();
    let expected = native
        .iter()
        .cloned()
        .map(|value| ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value,
        })
        .collect::<Vec<_>>();
    assert_eq!(response.message.content, expected);

    let encoded = encode(response.message.content).unwrap();
    assert_eq!(encoded["messages"][0]["content"], json!(native));
}

#[test]
fn mcp_tool_use_result_and_listing_decode_and_replay_as_native_blocks() {
    let native = vec![
        json!({
            "type":"mcp_tool_listing",
            "mcp_server_name":"calendar",
            "tools":[{
                "name":"find_event",
                "description":"Find an event",
                "input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
                "provider_extension":{"keep":true}
            }],
            "listing_extension":{"revision":2}
        }),
        json!({
            "type":"mcp_tool_use",
            "id":"mcptoolu_01MCP",
            "name":"find_event",
            "server_name":"calendar",
            "input":{"query":"release planning"},
            "provider_extension":{"keep":"the server identity"}
        }),
        json!({
            "type":"mcp_tool_result",
            "tool_use_id":"mcptoolu_01MCP",
            "is_error":false,
            "content":[{"type":"text","text":"Found one event"}],
            "result_extension":{"keep":7}
        }),
    ];
    let body = json!({
        "model":"claude-opus-5-5",
        "stop_reason":"end_turn",
        "content":native
    });
    let response = AnthropicMessagesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            },
            &wire_api::decode_context(),
        )
        .unwrap();
    assert_eq!(
        response.message.content,
        native
            .iter()
            .cloned()
            .map(|value| ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            })
            .collect::<Vec<_>>()
    );

    let encoded = encode(response.message.content).unwrap();
    assert_eq!(encoded["messages"][0]["content"], json!(native));
}

#[test]
fn streamed_server_tool_input_is_assembled_without_becoming_a_client_tool_call() {
    let start = json!({
        "type":"content_block_start",
        "index":0,
        "content_block":{
            "type":"server_tool_use",
            "id":"srvtoolu_web_fetch_1",
            "name":"web_fetch",
            "input":{},
            "provider_extension":{"kept":true}
        }
    });
    let empty_delta = json!({
        "type":"content_block_delta",
        "index":0,
        "delta":{"type":"input_json_delta","partial_json":""}
    });
    let frames = vec![
        json!({"type":"message_start","message":{"model":"claude-sonnet-4-5"}}),
        start.clone(),
        empty_delta.clone(),
        empty_delta.clone(),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"url\":"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"https://example.test\"}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ];
    let events = decode_stream(&frames).unwrap();

    let native = events.iter().find_map(|event| match event {
        StreamEvent::ProviderContent {
            block: 0, value, ..
        } => Some(value),
        _ => None,
    });
    assert_eq!(
        native,
        Some(&json!({
            "type":"server_tool_use",
            "id":"srvtoolu_web_fetch_1",
            "name":"web_fetch",
            "input":{"url":"https://example.test"},
            "provider_extension":{"kept":true}
        }))
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));

    let observed = provider_events(&events);
    assert_eq!(observed[0], &start);
    assert_eq!(observed[1], &empty_delta);
    assert_eq!(observed[2], &empty_delta);
    assert_eq!(observed[5]["type"], "content_block_stop");
    assert_eq!(observed.len(), 6, "repeated frames remain repeated");
}

#[test]
fn streamed_mcp_tool_input_is_assembled_with_its_server_identity() {
    let start = json!({
        "type":"content_block_start",
        "index":12,
        "content_block":{
            "type":"mcp_tool_use",
            "id":"mcptoolu_calendar_1",
            "name":"find_event",
            "server_name":"calendar",
            "input":{},
            "provider_extension":{"kept":true}
        }
    });
    let empty_delta = json!({
        "type":"content_block_delta",
        "index":12,
        "delta":{"type":"input_json_delta","partial_json":""}
    });
    let frames = vec![
        json!({"type":"message_start","message":{"model":"claude-opus-5-5"}}),
        start.clone(),
        empty_delta.clone(),
        empty_delta.clone(),
        json!({"type":"content_block_delta","index":12,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}),
        json!({"type":"content_block_delta","index":12,"delta":{"type":"input_json_delta","partial_json":"\"release planning\"}"}}),
        json!({"type":"content_block_stop","index":12}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ];
    let events = decode_stream(&frames).unwrap();

    assert_eq!(
        events.iter().find_map(|event| match event {
            StreamEvent::ProviderContent {
                block: 12, value, ..
            } => Some(value),
            _ => None,
        }),
        Some(&json!({
            "type":"mcp_tool_use",
            "id":"mcptoolu_calendar_1",
            "name":"find_event",
            "server_name":"calendar",
            "input":{"query":"release planning"},
            "provider_extension":{"kept":true}
        }))
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));

    let observed = provider_events(&events);
    assert_eq!(observed[0], &start);
    assert_eq!(observed[1], &empty_delta);
    assert_eq!(observed[2], &empty_delta);
    assert_eq!(observed[5]["type"], "content_block_stop");
    assert_eq!(observed.len(), 6, "repeated input frames remain repeated");
}

#[test]
fn unknown_block_deltas_remain_events_and_suppress_incomplete_replay() {
    let start = json!({
        "type":"content_block_start",
        "index":4,
        "content_block":{"type":"future_native_block","seed":{"a":1}}
    });
    let delta = json!({
        "type":"content_block_delta",
        "index":4,
        "delta":{"type":"future_native_delta","value":"opaque"}
    });
    let stop = json!({"type":"content_block_stop","index":4});
    let events = decode_stream(&[
        json!({"type":"message_start","message":{"model":"m"}}),
        start.clone(),
        delta.clone(),
        delta.clone(),
        stop.clone(),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ])
    .unwrap();

    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ProviderContent { block: 4, .. })));
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    let observed = provider_events(&events);
    assert_eq!(&observed[..4], &[&start, &delta, &delta, &stop]);
}

#[test]
fn complete_unknown_stream_blocks_are_replayable_only_after_block_stop() {
    let block = json!({
        "type":"new_server_result",
        "opaque":{"result_id":"r1","values":["a","b"]}
    });
    let events = decode_stream(&[
        json!({"type":"message_start","message":{"model":"m"}}),
        json!({"type":"content_block_start","index":2,"content_block":block.clone()}),
        json!({"type":"content_block_stop","index":2}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ])
    .unwrap();
    assert_eq!(
        events.iter().find_map(|event| match event {
            StreamEvent::ProviderContent {
                block: 2, value, ..
            } => Some(value),
            _ => None,
        }),
        Some(&block)
    );
}

#[test]
fn unknown_events_are_observational_and_native_citation_frames_remain_available() {
    let citation = json!({
        "type":"content_block_delta",
        "index":0,
        "delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://example.test","title":"Example"}}
    });
    let events = decode_stream(&[
        json!({"type":"message_start","message":{"model":"m"}}),
        json!({"type":"provider_event_added_later","payload":{"x":1}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"cited"}}),
        citation.clone(),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ])
    .unwrap();

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ProviderEvent { payload, .. }
            if payload["type"] == "provider_event_added_later"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::TextDelta { text, .. } if text == "cited"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ProviderEvent { payload, .. } if payload == &citation
    )));
    assert!(events
        .iter()
        .any(|event| matches!(event, StreamEvent::WebSearch { .. })));
}

#[test]
fn fragmented_server_tool_input_obeys_the_aggregate_buffer_limit() {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"message_start","message":{"model":"m"}}"#,
    )
    .unwrap();
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","name":"web_fetch","input":{}}}"#,
    )
    .unwrap();

    let fragment = "x".repeat(4 * 1024 * 1024);
    for _ in 0..2 {
        let frame = json!({
            "type":"content_block_delta",
            "index":0,
            "delta":{"type":"input_json_delta","partial_json":fragment}
        });
        let result = wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes());
        if result.is_err() {
            assert!(matches!(result, Err(LlmError::StreamInterrupted { .. })));
            return;
        }
    }
    panic!("server tool input exceeded the aggregate pending limit without failing");
}

#[test]
fn fragmented_mcp_tool_input_obeys_the_aggregate_buffer_limit() {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"message_start","message":{"model":"m"}}"#,
    )
    .unwrap();
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"content_block_start","index":0,"content_block":{"type":"mcp_tool_use","id":"mcptoolu_1","name":"echo","server_name":"calendar","input":{}}}"#,
    )
    .unwrap();

    let fragment = "x".repeat(4 * 1024 * 1024);
    for _ in 0..2 {
        let frame = json!({
            "type":"content_block_delta",
            "index":0,
            "delta":{"type":"input_json_delta","partial_json":fragment}
        });
        let result = wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes());
        if result.is_err() {
            assert!(matches!(result, Err(LlmError::StreamInterrupted { .. })));
            return;
        }
    }
    panic!("MCP tool input exceeded the aggregate pending limit without failing");
}

#[test]
fn empty_or_non_object_server_tool_json_is_not_replayed_as_complete_input() {
    for (index, partial_json) in [(0, ""), (1, "null")] {
        let events = decode_stream(&[
            json!({"type":"message_start","message":{"model":"m"}}),
            json!({
                "type":"content_block_start",
                "index":index,
                "content_block":{"type":"server_tool_use","id":"srv","name":"web_fetch","input":{}}
            }),
            json!({
                "type":"content_block_delta",
                "index":index,
                "delta":{"type":"input_json_delta","partial_json":partial_json}
            }),
            json!({"type":"content_block_stop","index":index}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
            json!({"type":"message_stop"}),
        ])
        .unwrap();
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::ProviderContent { block, .. } if *block == index
        )));
        assert!(provider_events(&events).iter().any(|payload| {
            payload["type"] == "content_block_delta" && payload["index"] == index
        }));
    }
}

#[test]
fn malformed_and_unknown_mcp_input_deltas_suppress_replay_but_keep_frames() {
    for (index, partial_json) in [(20, ""), (21, "null"), (22, "{\"query\":")] {
        let start = json!({
            "type":"content_block_start",
            "index":index,
            "content_block":{
                "type":"mcp_tool_use",
                "id":"mcptoolu_bad",
                "name":"find_event",
                "server_name":"calendar",
                "input":{}
            }
        });
        let delta = json!({
            "type":"content_block_delta",
            "index":index,
            "delta":{"type":"input_json_delta","partial_json":partial_json}
        });
        let events = decode_stream(&[
            json!({"type":"message_start","message":{"model":"m"}}),
            start.clone(),
            delta.clone(),
            json!({"type":"content_block_stop","index":index}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
            json!({"type":"message_stop"}),
        ])
        .unwrap();
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::ProviderContent { block, .. } if *block == index
        )));
        assert!(provider_events(&events)
            .iter()
            .any(|payload| { *payload == &start }));
        assert!(provider_events(&events)
            .iter()
            .any(|payload| { *payload == &delta }));
        assert!(!events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    }

    let start = json!({
        "type":"content_block_start",
        "index":23,
        "content_block":{
            "type":"mcp_tool_use",
            "id":"mcptoolu_unknown",
            "name":"find_event",
            "server_name":"calendar",
            "input":{}
        }
    });
    let unknown_delta = json!({
        "type":"content_block_delta",
        "index":23,
        "delta":{"type":"future_mcp_delta","opaque":{"v":1}}
    });
    let stop = json!({"type":"content_block_stop","index":23});
    let events = decode_stream(&[
        json!({"type":"message_start","message":{"model":"m"}}),
        start.clone(),
        unknown_delta.clone(),
        stop.clone(),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ])
    .unwrap();
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ProviderContent { block: 23, .. })));
    assert_eq!(
        &provider_events(&events)[..3],
        &[&start, &unknown_delta, &stop]
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
}

#[test]
fn mcp_input_delta_without_a_string_fragment_suppresses_replay() {
    let start = json!({
        "type":"content_block_start",
        "index":24,
        "content_block":{
            "type":"mcp_tool_use",
            "id":"mcptoolu_missing_fragment",
            "name":"find_event",
            "server_name":"calendar",
            "input":{}
        }
    });
    let malformed_delta = json!({
        "type":"content_block_delta",
        "index":24,
        "delta":{"type":"input_json_delta","partial_json":null}
    });
    let events = decode_stream(&[
        json!({"type":"message_start","message":{"model":"m"}}),
        start.clone(),
        malformed_delta.clone(),
        json!({"type":"content_block_stop","index":24}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ])
    .unwrap();

    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ProviderContent { block: 24, .. })));
    assert_eq!(&provider_events(&events)[..2], &[&start, &malformed_delta]);
}

#[test]
fn an_open_mcp_input_block_fails_at_stream_termination() {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"message_start","message":{"model":"m"}}"#,
    )
    .unwrap();
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"content_block_start","index":31,"content_block":{"type":"mcp_tool_use","id":"mcptoolu_open","name":"echo","server_name":"calendar","input":{}}}"#,
    )
    .unwrap();
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"content_block_delta","index":31,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}"#,
    )
    .unwrap();

    let terminal = wire_api::decode_frame(&mut *decoder, br#"{"type":"message_stop"}"#);
    assert!(matches!(terminal, Err(LlmError::StreamInterrupted { .. })));
}

#[test]
fn citations_arriving_midstream_reconstruct_the_complete_text_block() {
    let first_citation = json!({
        "type":"char_location",
        "cited_text":"A supported passage.",
        "document_index":0,
        "document_title":"Notes",
        "start_char_index":0,
        "end_char_index":20
    });
    let second_citation = json!({
        "type":"page_location",
        "cited_text":"Another supported passage.",
        "document_index":1,
        "document_title":"Report",
        "start_page_number":4,
        "end_page_number":5
    });
    let frames = vec![
        json!({"type":"message_start","message":{"model":"m"}}),
        json!({
            "type":"content_block_start",
            "index":3,
            "content_block":{
                "type":"text",
                "text":"According to ",
                "provider_metadata":{"keep":"this"}
            }
        }),
        json!({"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"the report, "}}),
        json!({"type":"content_block_delta","index":3,"delta":{"type":"citations_delta","citation":first_citation.clone()}}),
        json!({"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"this result follows."}}),
        json!({"type":"content_block_delta","index":3,"delta":{"type":"citations_delta","citation":second_citation.clone()}}),
        json!({"type":"content_block_stop","index":3}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ];
    let events = decode_stream(&frames).unwrap();

    let text = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextDelta { block: 3, text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        text,
        ["According to ", "the report, ", "this result follows."]
    );
    assert_eq!(
        text.concat(),
        "According to the report, this result follows."
    );
    assert_eq!(
        events.iter().find_map(|event| match event {
            StreamEvent::ProviderContent {
                block: 3, value, ..
            } => Some(value),
            _ => None,
        }),
        Some(&json!({
            "type":"text",
            "text":"According to the report, this result follows.",
            "citations":[first_citation, second_citation],
            "provider_metadata":{"keep":"this"}
        }))
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::WebSearch { .. })));

    let observed = provider_events(&events);
    assert_eq!(
        observed
            .iter()
            .map(|frame| frame["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop"
        ]
    );
}

#[test]
fn plain_streamed_text_includes_its_start_prefix_without_native_content() {
    let events = decode_stream(&[
        json!({"type":"message_start","message":{"model":"m"}}),
        json!({"type":"content_block_start","index":9,"content_block":{"type":"text","text":"Hello"}}),
        json!({"type":"content_block_delta","index":9,"delta":{"type":"text_delta","text":" world"}}),
        json!({"type":"content_block_stop","index":9}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ])
    .unwrap();
    let text = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextDelta { block: 9, text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(text, "Hello world");
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ProviderContent { block: 9, .. })));
}

#[test]
fn malformed_citations_and_unknown_text_deltas_preserve_frames_without_replay() {
    let start = json!({
        "type":"content_block_start",
        "index":5,
        "content_block":{"type":"text","text":"prefix"}
    });
    let text_delta = json!({
        "type":"content_block_delta",
        "index":5,
        "delta":{"type":"text_delta","text":" middle"}
    });
    let malformed_citation = json!({
        "type":"content_block_delta",
        "index":5,
        "delta":{"type":"citations_delta","citation":null}
    });
    let unknown_delta = json!({
        "type":"content_block_delta",
        "index":5,
        "delta":{"type":"future_text_delta","opaque":true}
    });
    let stop = json!({"type":"content_block_stop","index":5});
    let events = decode_stream(&[
        json!({"type":"message_start","message":{"model":"m"}}),
        start.clone(),
        text_delta.clone(),
        malformed_citation.clone(),
        unknown_delta.clone(),
        stop.clone(),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ])
    .unwrap();

    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ProviderContent { block: 5, .. })));
    assert_eq!(
        &provider_events(&events)[..5],
        &[
            &start,
            &text_delta,
            &malformed_citation,
            &unknown_delta,
            &stop
        ]
    );
}

#[test]
fn malformed_initial_citation_text_is_observed_but_not_replayed() {
    for (index, block) in [
        (6, json!({"type":"text","citations":[]})),
        (
            7,
            json!({"type":"text","text":"answer","citations":"invalid"}),
        ),
    ] {
        let events = decode_stream(&[
            json!({"type":"message_start","message":{"model":"m"}}),
            json!({"type":"content_block_start","index":index,"content_block":block}),
            json!({"type":"content_block_stop","index":index}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
            json!({"type":"message_stop"}),
        ])
        .unwrap();
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::ProviderContent { block, .. } if *block == index
        )));
        assert!(provider_events(&events)
            .iter()
            .any(|frame| frame["type"] == "content_block_start"));
    }
}

#[test]
fn incomplete_cited_text_stream_fails_without_emitting_replay_content() {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    let frames = [
        json!({"type":"message_start","message":{"model":"m"}}),
        json!({"type":"content_block_start","index":8,"content_block":{"type":"text","text":"hello"}}),
        json!({"type":"content_block_delta","index":8,"delta":{"type":"citations_delta","citation":{"type":"char_location"}}}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
    ];
    let mut events = Vec::new();
    for frame in frames {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap());
    }
    let end = wire_api::decode_frame(&mut *decoder, br#"{"type":"message_stop"}"#);
    assert!(matches!(end, Err(LlmError::StreamInterrupted { .. })));
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ProviderContent { block: 8, .. })));
}

#[test]
fn cited_text_buffer_is_bounded_and_released_after_plain_text_blocks() {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"message_start","message":{"model":"m"}}"#,
    )
    .unwrap();
    let large_delta = "x".repeat(4 * 1024 * 1024 + 128);
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
    )
    .unwrap();
    let oversized = json!({
        "type":"content_block_delta",
        "index":0,
        "delta":{"type":"text_delta","text":large_delta}
    });
    assert!(matches!(
        wire_api::decode_frame(&mut *decoder, oversized.to_string().as_bytes()),
        Err(LlmError::StreamInterrupted { .. })
    ));

    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"message_start","message":{"model":"m"}}"#,
    )
    .unwrap();
    let large_delta = "y".repeat(2 * 1024 * 1024 + 128);
    for index in 0..2 {
        wire_api::decode_frame(
            &mut *decoder,
            json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}})
                .to_string()
                .as_bytes(),
        )
        .unwrap();
        wire_api::decode_frame(
            &mut *decoder,
            json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":large_delta}})
                .to_string()
                .as_bytes(),
        )
        .unwrap();
        wire_api::decode_frame(
            &mut *decoder,
            json!({"type":"content_block_stop","index":index})
                .to_string()
                .as_bytes(),
        )
        .unwrap();
    }
    wire_api::decode_frame(
        &mut *decoder,
        br#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
    )
    .unwrap();
    wire_api::decode_frame(&mut *decoder, br#"{"type":"message_stop"}"#).unwrap();
}
