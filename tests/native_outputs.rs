use lingxi_llm_client::{protocol::*, *};
use serde_json::json;
fn context() -> CodecContext {
    let p:ProviderProfile=serde_json::from_value(json!({"provider_id":"openai","profile_name":"openai","protocol":"open_ai_responses","base_url":"https://test.invalid","auth":"none"})).unwrap();
    CodecContext::new(&p, "test", RequestMode::Complete)
}
#[test]
fn hosted_and_unknown_items_survive_buffered_and_fragmented_streaming_once() {
    for kind in [
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "future_hosted_item",
    ] {
        let item = json!({"id":"native-id","type":kind,"status":"completed","native_data":{"important":true}});
        let body = json!({"id":"resp-test","status":"completed","model":"test","output":[item]});
        let response = OpenAiResponsesCodec
            .decode_response(
                &HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&body).unwrap().into(),
                },
                &context(),
            )
            .unwrap();
        assert_eq!(
            response.message.content[0],
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::OpenAiResponses,
                value: item.clone()
            }
        );
        if kind == "mcp_approval_request" {
            assert_eq!(
                response.stop_reason,
                StopReason::Other("requires_action".into())
            );
        }
        let wire = format!(
            "data: {}\n\ndata: {}\n\n",
            json!({"type":"response.output_item.done","output_index":0,"item":item}),
            json!({"type":"response.completed","response":body})
        );
        for width in [1, 7, wire.len()] {
            let mut decoder = OpenAiResponsesCodec.stream_decoder(&context());
            let mut events = vec![];
            for chunk in wire.as_bytes().chunks(width) {
                events.extend(
                    decoder
                        .push_bytes(chunk)
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                );
            }
            events.extend(
                decoder
                    .finish()
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
            );
            let native = events
                .iter()
                .filter_map(|e| match e {
                    StreamEvent::ProviderContent { value, .. } => Some(value),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(native, vec![&item]);
            if kind == "mcp_approval_request" {
                assert!(events.iter().any(|e|matches!(e,StreamEvent::End{stop_reason:StopReason::Other(reason),..} if reason=="requires_action")));
            }
        }
    }
}
