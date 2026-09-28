use lingxi_llm_client::providers::openai::types::{
    CodeInterpreterConfig, CodeInterpreterMemoryLimit,
};
#[path = "support/wire_api.rs"]
mod wire_api;

use lingxi_llm_client::{
    codecs::{openai::responses::OpenAiResponsesCodec, EncodeRequest, WireCodec},
    protocol::{
        ChatRequest, ContentBlock, LlmError, ProtocolFamily, ProviderProfile, Region, StreamEvent,
    },
    providers::openai::containers::{OpenAiContainerRef, OpenAiContainerScope},
    HttpResponse, LlmClientBuilder, RequestOptions,
};
use serde_json::{json, Value};
use std::sync::Arc;

mod support;

fn profile(model: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"qwen",
        "profile_name":"qwen-search-test",
        "protocol":"open_ai_responses",
        "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        "auth":"none",
        "extra":{"web_search":"qwen","file_search":"qwen"},
        "models":[{"display_model":model,"request_model":model,"billing_model":model}]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model":"qwen3.8-max",
        "messages":[{"role":"user","content":[{"type":"text","text":"Calculate 2 + 2"}]}]
    }))
    .unwrap();
    request.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::CodeInterpreter(
            CodeInterpreterConfig {
                memory_limit: None,
                ..CodeInterpreterConfig::default()
            },
        )
        .into(),
    );
    request
}

#[test]
fn qwen_responses_encodes_documented_tool_and_required_thinking_flag() {
    let profile = profile("qwen3.8-max");
    let request = request();
    let context = wire_api::context(&profile, "qwen3.8-max", &RequestOptions::default());
    let outgoing = OpenAiResponsesCodec
        .encode_request(EncodeRequest::new(&request), &context)
        .unwrap();
    let body: Value = serde_json::from_slice(&outgoing.body).unwrap();

    assert_eq!(body["tools"], json!([{"type":"code_interpreter"}]));
    assert_eq!(body["enable_thinking"], true);
    assert!(body.get("include").is_none());
    assert!(body["tools"][0].get("container").is_none());
}

#[tokio::test]
async fn undocumented_combinations_are_rejected_before_transport() {
    let profile = profile("qwen3.8-max");
    let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();

    let mut with_memory = request();
    with_memory.hosted_tools = vec![
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::CodeInterpreter(
            CodeInterpreterConfig {
                memory_limit: Some(CodeInterpreterMemoryLimit::FourG),
                ..CodeInterpreterConfig::default()
            },
        )
        .into(),
    ];

    let mut with_function = request();
    with_function.tools = serde_json::from_value(json!([{
        "name":"local_function",
        "description":"A caller-owned function",
        "input_schema":{"type":"object"}
    }]))
    .unwrap();

    let mut forced_choice = request();
    forced_choice.tool_choice = serde_json::from_value(json!({"type":"none"})).unwrap();

    let mut thinking_disabled = request();
    thinking_disabled.thinking = serde_json::from_value(json!({"mode":"disabled"})).unwrap();

    for request in [with_memory, with_function, forced_choice, thinking_disabled] {
        let result = client
            .chat()
            .complete(&request, &RequestOptions::default())
            .await;
        assert!(
            matches!(
                result,
                Err(LlmError::InvalidRequest { .. } | LlmError::UnsupportedCapability { .. })
            ),
            "expected local preflight error, got {result:?}"
        );
    }
}

#[tokio::test]
async fn qwen_rejects_openai_container_references_before_transport() {
    let profile = profile("qwen3.8-max");
    let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    let scope = OpenAiContainerScope::new("qwen-search-test", "account-a").unwrap();
    let container = OpenAiContainerRef::from_id(&scope, "cntr_openai-only").unwrap();
    let mut request = request();
    request.hosted_tools = vec![
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::CodeInterpreter(
            CodeInterpreterConfig::default().with_container(container),
        )
        .into(),
    ];

    assert!(matches!(
        client
            .chat()
            .complete(&request, &RequestOptions::default())
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[tokio::test]
async fn unsupported_model_and_conflicting_profile_defaults_are_preflighted() {
    let unsupported_profile = profile("qwen3.7-max");
    let unsupported_client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[unsupported_profile])
            .with_region(Region::International)
            .build()
            .unwrap();
    let mut unsupported_request = request();
    unsupported_request.model = "qwen3.7-max".into();
    assert!(matches!(
        unsupported_client
            .chat()
            .complete(&unsupported_request, &RequestOptions::default())
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut conflicting_profile = profile("qwen3.8-max");
    conflicting_profile.extra["body"]["enable_thinking"] = Value::Bool(false);
    let conflicting_client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[conflicting_profile])
            .with_region(Region::International)
            .build()
            .unwrap();
    assert!(matches!(
        conflicting_client
            .chat()
            .complete(&request(), &RequestOptions::default())
            .await,
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn qwen_response_preserves_call_output_and_provider_tool_usage() {
    let profile = profile("qwen3.8-max");
    let context = wire_api::context(&profile, "qwen3.8-max", &RequestOptions::default());
    let item = json!({
        "id":"ci-1",
        "type":"code_interpreter_call",
        "status":"completed",
        "container_id":"cntr-1",
        "code":"print(2 + 2)",
        "outputs":[{"type":"logs","logs":"4"}]
    });
    let response = OpenAiResponsesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&json!({
                    "id":"resp-1",
                    "status":"completed",
                    "model":"qwen3.8-max",
                    "output":[item],
                    "usage":{
                        "input_tokens":8,
                        "output_tokens":5,
                        "total_tokens":13,
                        "x_tools":{"code_interpreter":{"count":2}}
                    }
                }))
                .unwrap()
                .into(),
            },
            &context,
        )
        .unwrap();

    assert_eq!(
        response.message.content,
        vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: item,
        }]
    );
    assert_eq!(
        response
            .usage
            .usage
            .and_then(|usage| usage.server_tool_usage)
            .and_then(|usage| usage.code_interpreter_requests),
        Some(2)
    );
}

#[test]
fn qwen_stream_keeps_status_events_and_complete_native_output() {
    let profile = profile("qwen3.8-max");
    let options = wire_api::EncodingOptions {
        stream: true,
        file_account_scope: None,
    };
    let context = wire_api::context(&profile, "qwen3.8-max", &options);
    let item = json!({
        "id":"ci-1",
        "type":"code_interpreter_call",
        "status":"completed",
        "container_id":"cntr-1",
        "code":"print(2 + 2)",
        "outputs":[{"type":"logs","logs":"4"}]
    });
    let completed = json!({
        "id":"resp-1",
        "status":"completed",
        "model":"qwen3.8-max",
        "output":[item],
        "usage":{
            "input_tokens":8,
            "output_tokens":5,
            "total_tokens":13,
            "x_tools":{"code_interpreter":{"count":2}}
        }
    });
    let frames = [
        json!({
            "type":"response.code_interpreter_call.in_progress",
            "sequence_number":3,
            "output_index":0,
            "item_id":"ci-1"
        }),
        json!({
            "type":"response.code_interpreter_call.interpreting",
            "sequence_number":4,
            "output_index":0,
            "item_id":"ci-1"
        }),
        json!({
            "type":"response.code_interpreter_call.completed",
            "sequence_number":5,
            "output_index":0,
            "item_id":"ci-1"
        }),
        json!({
            "type":"response.output_item.done",
            "sequence_number":6,
            "output_index":0,
            "item":item
        }),
        json!({"type":"response.completed","sequence_number":7,"response":completed}),
    ];
    let stream = frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&context);
    let mut events = Vec::new();
    for chunk in stream.as_bytes().chunks(13) {
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
        .filter_map(|event| match event {
            StreamEvent::ProviderContent { value, .. } => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(native.len(), 4);
    assert_eq!(
        native[..3]
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call.completed"
        ]
    );
    assert_eq!(native[3], &item);
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    assert!(events.iter().any(|event| matches!(event,
        StreamEvent::End { usage, .. }
            if usage.usage.and_then(|usage| usage.server_tool_usage)
                .and_then(|usage| usage.code_interpreter_requests) == Some(2)
    )));
}
