#[path = "support/wire_api.rs"]
mod wire_api;
use lingxi_llm_client::protocol::{ContentBlock, StopReason, StreamEvent};
use lingxi_llm_client::{
    AnthropicMessagesCodec, GeminiCodec, HttpResponse, OpenAiChatCodec, OpenAiResponsesCodec,
    WireCodec,
};
use serde_json::{json, Value};

fn response(body: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![],
        body: body.to_string().into(),
    }
}
fn annotation() -> Value {
    json!({"type":"url_citation","url":"https://example.com","title":"Source","start_index":0,"end_index":4})
}
fn searches(events: &[StreamEvent]) -> Vec<&lingxi_llm_client::protocol::WebSearchResult> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::WebSearch { result } => Some(result),
            _ => None,
        })
        .collect()
}

#[test]
fn responses_preserves_annotations_search_actions_and_local_tools() {
    let a = annotation();
    let call = json!({"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","queries":["news"],"sources":[{"url":"https://example.com"}]}});
    let result = OpenAiResponsesCodec.decode_response(&response(json!({"output":[call,{"type":"message","content":[{"type":"output_text","text":"News","annotations":[a]}]},{"type":"function_call","call_id":"local","name":"save","arguments":"{}"}]})), &wire_api::decode_context()).unwrap();
    let search = result.web_search.unwrap();
    assert_eq!(search.citations[0].url, "https://example.com");
    assert_eq!(search.metadata["annotations"][0]["annotation"], a);
    assert_eq!(search.metadata["web_search_calls"][0], call);
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(
        result
            .message
            .content
            .iter()
            .filter(|c| matches!(c, ContentBlock::ToolUse { .. }))
            .count(),
        1
    );
}

#[test]
fn responses_stream_deduplicates_terminal_annotations_and_calls() {
    let a = annotation();
    let call = json!({"type":"web_search_call","id":"ws","status":"completed"});
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let frames = [
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws"}}),
        json!({"type":"response.output_item.done","output_index":0,"item":call}),
        json!({"type":"response.output_text.annotation.added","output_index":1,"content_index":0,"annotation":a}),
        json!({"type":"response.completed","response":{"output":[call,{"type":"message","content":[{"type":"output_text","annotations":[a]}]}]}}),
    ];
    let events: Vec<_> = frames
        .iter()
        .flat_map(|f| wire_api::decode_frame(&mut *decoder, f.to_string().as_bytes()).unwrap())
        .collect();
    assert_eq!(searches(&events).len(), 2);
    assert!(!events
        .iter()
        .any(|e| matches!(e, StreamEvent::ToolCallDelta { .. })));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::EndTurn,
            ..
        })
    ));
}

#[test]
fn responses_terminal_only_annotations_are_not_lost() {
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let events = wire_api::decode_frame(&mut *decoder , json!({"type":"response.completed","response":{"output":[{"type":"message","content":[{"annotations":[annotation()]}]}]}}).to_string().as_bytes()).unwrap();
    assert_eq!(searches(&events)[0].citations.len(), 1);
}

#[test]
fn responses_top_level_server_usage_survives_complete_and_stream() {
    let counts = json!({"SERVER_SIDE_TOOL_WEB_SEARCH": 2});
    let body = json!({"output": [], "server_side_tool_usage": counts});
    let decoded = OpenAiResponsesCodec
        .decode_response(&response(body.clone()), &wire_api::decode_context())
        .unwrap();
    assert_eq!(
        decoded.web_search.unwrap().metadata["server_side_tool_usage"],
        counts
    );
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let frame = json!({"type": "response.completed", "response": body});
    let events = wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap();
    assert_eq!(
        searches(&events)[0].metadata["server_side_tool_usage"],
        counts
    );
}

#[test]
fn chat_and_openrouter_nested_annotations_survive_streaming() {
    let a = json!({"type":"url_citation","url_citation":{"url":"https://example.com","title":"Source","content":"excerpt","start_index":0,"end_index":4}});
    let result = OpenAiChatCodec.decode_response(&response(json!({"choices":[{"message":{"content":"News","annotations":[a]},"finish_reason":"stop"}]})), &wire_api::decode_context()).unwrap();
    assert_eq!(result.web_search.unwrap().metadata["annotations"][0], a);
    let mut decoder = OpenAiChatCodec.stream_decoder(&wire_api::decode_context());
    let frame = json!({"choices":[{"delta":{"annotations":[a]}}]}).to_string();
    assert_eq!(
        searches(&wire_api::decode_frame(&mut *decoder, frame.as_bytes()).unwrap())[0]
            .citations
            .len(),
        1
    );
    assert!(searches(&wire_api::decode_frame(&mut *decoder, frame.as_bytes()).unwrap()).is_empty());
}

#[test]
fn anthropic_search_errors_and_encrypted_attribution_are_preserved() {
    let cited = json!({"type":"text","text":"News","citations":[{"type":"web_search_result_location","url":"https://example.com","title":"Source","encrypted_index":"secret","cited_text":"News"}]});
    let error = json!({"type":"web_search_tool_result","tool_use_id":"srv_1","content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}});
    let result = AnthropicMessagesCodec
        .decode_response(
            &response(json!({"content":[error,cited],"stop_reason":"end_turn"})),
            &wire_api::decode_context(),
        )
        .unwrap();
    let search = result.web_search.unwrap();
    assert_eq!(search.citations.len(), 1);
    assert_eq!(search.metadata["content"], json!([error, cited]));
    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert!(!result
        .message
        .content
        .iter()
        .any(|c| matches!(c, ContentBlock::ToolUse { .. })));
    let error_only = AnthropicMessagesCodec
        .decode_response(
            &response(json!({"content":[error]})),
            &wire_api::decode_context(),
        )
        .unwrap()
        .web_search
        .unwrap();
    assert!(error_only.citations.is_empty());
}

#[test]
fn anthropic_stream_server_arguments_are_not_client_tool_calls_or_deduplicated() {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    let frames = [
        json!({"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv","name":"web_search","input":{}}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"a"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"a"}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://example.com","encrypted_index":"secret"}}}),
    ];
    let events: Vec<_> = frames
        .iter()
        .flat_map(|f| wire_api::decode_frame(&mut *decoder, f.to_string().as_bytes()).unwrap())
        .collect();
    assert_eq!(searches(&events).len(), 4);
    assert!(!events
        .iter()
        .any(|e| matches!(e, StreamEvent::ToolCallDelta { .. })));
    assert_eq!(searches(&events)[3].citations[0].url, "https://example.com");
}

#[test]
fn gemini_preserves_search_widget_supports_and_metadata_only_frames() {
    let metadata = json!({"webSearchQueries":["news"],"searchEntryPoint":{"renderedContent":"<div>Search</div>"},"groundingChunks":[{"web":{"uri":"https://example.com","title":"Source"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":4},"groundingChunkIndices":[0]}]});
    let result = GeminiCodec
        .decode_response(
            &response(json!({"candidates":[{"groundingMetadata":metadata}]})),
            &wire_api::decode_context(),
        )
        .unwrap();
    let search = result.web_search.unwrap();
    assert_eq!(search.citations.len(), 1);
    assert_eq!(search.metadata["groundingMetadata"], metadata);
    let mut decoder = GeminiCodec.stream_decoder(&wire_api::decode_context());
    let events = wire_api::decode_frame(
        &mut *decoder,
        json!({"candidates":[{"groundingMetadata":{"webSearchQueries":["news"]}}]})
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    assert!(searches(&events)[0].citations.is_empty());
    assert_eq!(
        searches(&events)[0].metadata["groundingMetadata"]["webSearchQueries"],
        json!(["news"])
    );
}

#[test]
fn ordinary_responses_do_not_claim_web_search() {
    assert!(OpenAiChatCodec
        .decode_response(
            &response(json!({"choices":[{"message":{"content":"hello"}}]})),
            &wire_api::decode_context()
        )
        .unwrap()
        .web_search
        .is_none());
    assert!(OpenAiResponsesCodec
        .decode_response(&response(json!({"output":[]})), &wire_api::decode_context())
        .unwrap()
        .web_search
        .is_none());
    assert!(AnthropicMessagesCodec
        .decode_response(
            &response(json!({"content":[{"type":"text","text":"hello"}]})),
            &wire_api::decode_context()
        )
        .unwrap()
        .web_search
        .is_none());
    assert!(GeminiCodec
        .decode_response(
            &response(json!({"candidates":[{}]})),
            &wire_api::decode_context()
        )
        .unwrap()
        .web_search
        .is_none());
}

#[test]
fn anthropic_pause_turn_and_citations_replay_exact_native_blocks() {
    use lingxi_llm_client::protocol::{
        CompletionRequest, FailoverTriggers, ModelCapabilitySupport, ProviderId, ProviderProfile,
        ToolChoice,
    };
    use lingxi_llm_client::{PricingModelRef, RequestOptions, ResolvedRoute};
    let blocks = json!([
        {"type":"server_tool_use","id":"srv","name":"web_search","input":{"query":"news"}},
        {"type":"web_search_tool_result","tool_use_id":"srv","content":[{"type":"web_search_result","url":"https://example.com","title":"Source","encrypted_content":"ciphertext","page_age":"today"}]},
        {"type":"text","text":"News","citations":[{"type":"web_search_result_location","url":"https://example.com","title":"Source","encrypted_index":"secret","cited_text":"News"}]}
    ]);
    let decoded = AnthropicMessagesCodec
        .decode_response(
            &response(json!({"content":blocks,"stop_reason":"pause_turn"})),
            &wire_api::decode_context(),
        )
        .unwrap();
    assert_eq!(decoded.stop_reason, StopReason::Other("pause_turn".into()));
    assert_eq!(decoded.message.text(), "News");
    let req = CompletionRequest {
        service_tier: None,
        model: "m".into(),
        web_search: None,
        file_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![decoded.message],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: None,
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    };
    let profile: ProviderProfile = serde_json::from_value(json!({"provider_id":"anthropic","profile_name":"anthropic","base_url":"https://api.anthropic.com","protocol":"anthropic_messages","auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m"}]})).unwrap();
    let route = ResolvedRoute {
        provider_id: ProviderId::new("anthropic"),
        profile_name: "anthropic".into(),
        request_model: "m".into(),
        display_model: "m".into(),
        pricing_model: PricingModelRef {
            pricing_provider_id: ProviderId::new("anthropic"),
            billing_model: "m".into(),
            request_model: "m".into(),
            display_model: "m".into(),
        },
        capability_support: ModelCapabilitySupport::default(),
        connection_chain: vec![],
        failover: FailoverTriggers::default(),
    };
    let encoded = AnthropicMessagesCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(&profile, &route.request_model, &RequestOptions::default()),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["messages"][0]["content"], blocks);
}

#[test]
fn native_search_counters_survive_full_and_usage_only_stream_frames() {
    let counters = json!({"web_search_requests":2,"web_fetch_requests":1});
    let decoded = AnthropicMessagesCodec
        .decode_response(
            &response(json!({"content":[],"usage":{"server_tool_use":counters}})),
            &wire_api::decode_context(),
        )
        .unwrap();
    assert_eq!(
        decoded.web_search.unwrap().metadata["usage"]["server_tool_use"],
        counters
    );
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    let events = wire_api::decode_frame(
        &mut *decoder,
        json!({"type":"message_delta","usage":{"server_tool_use":counters}})
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(
        searches(&events)[0].metadata["usage"]["server_tool_use"],
        counters
    );
    let mut decoder = OpenAiChatCodec.stream_decoder(&wire_api::decode_context());
    let events = wire_api::decode_frame(
        &mut *decoder,
        json!({"choices":[],"usage":{"server_tool_use":counters}})
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(
        searches(&events)[0].metadata["usage"]["server_tool_use"],
        counters
    );
    let decoded = OpenAiResponsesCodec
        .decode_response(
            &response(
                json!({"output":[],"usage":{"server_side_tool_usage":{"web_search_calls":3}}}),
            ),
            &wire_api::decode_context(),
        )
        .unwrap();
    assert_eq!(
        decoded.web_search.unwrap().metadata["usage"]["server_side_tool_usage"]["web_search_calls"],
        3
    );
}
