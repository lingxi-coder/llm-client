use lingxi_llm_client::providers::openai::types::CodeInterpreterConfig;
#[path = "support/wire_api.rs"]
mod wire_api;

use lingxi_llm_client::{
    codecs::{openai::responses::OpenAiResponsesCodec, EncodeRequest, WireCodec},
    protocol::{
        ChatRequest, ContentBlock, HostedTool, LlmError, ProtocolFamily, ProviderProfile, Region,
        StreamEvent, WebSearchConfig,
    },
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
        "extra":{"web_search":"qwen"},
        "models":[{"display_model":model,"request_model":model,"billing_model":model}]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model":"qwen3.8-max",
        "messages":[{"role":"user","content":[{"type":"text","text":"Summarize this page"}]}]
    }))
    .unwrap();
    request
        .hosted_tools
        .push(HostedTool::WebSearch(WebSearchConfig::default()));
    request
        .hosted_tools
        .push(lingxi_llm_client::providers::qwen::native::QwenHostedTool::WebExtractor.into());
    request
}

#[test]
fn qwen_responses_encodes_web_search_and_extractor_with_thinking() {
    let profile = profile("qwen3.8-max");
    let mut request = request();
    // Alibaba documents Code Interpreter as an optional companion tool.
    request.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::CodeInterpreter(
            CodeInterpreterConfig::default(),
        )
        .into(),
    );
    let context = wire_api::context(&profile, "qwen3.8-max", &RequestOptions::default());
    let outgoing = OpenAiResponsesCodec
        .encode_request(EncodeRequest::new(&request), &context)
        .unwrap();
    let body: Value = serde_json::from_slice(&outgoing.body).unwrap();

    assert_eq!(
        body["tools"],
        json!([
            {"type":"web_search"},
            {"type":"web_extractor"},
            {"type":"code_interpreter"}
        ])
    );
    assert_eq!(body["enable_thinking"], true);
    assert_eq!(body["tool_choice"], "auto");
}

#[tokio::test]
async fn extractor_without_search_and_disabled_thinking_are_rejected_locally() {
    let profile = profile("qwen3.8-max");
    let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();

    let mut missing_search = request();
    missing_search
        .hosted_tools
        .retain(|tool| !matches!(tool, HostedTool::WebSearch(_)));
    let mut thinking_disabled = request();
    thinking_disabled.thinking = serde_json::from_value(json!({"mode":"disabled"})).unwrap();

    for request in [missing_search, thinking_disabled] {
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
async fn unsupported_profiles_models_and_conflicting_defaults_are_rejected() {
    let mut unsupported_profile = profile("qwen3.8-max");
    unsupported_profile.provider_id = "other".into();
    let unsupported_client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[unsupported_profile])
            .with_region(Region::International)
            .build()
            .unwrap();
    assert!(matches!(
        unsupported_client
            .chat()
            .complete(&request(), &RequestOptions::default())
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let unsupported_model = profile("qwen3.7-max");
    let unsupported_client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[unsupported_model])
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
fn response_preserves_qwen_call_and_extractor_usage_count() {
    let profile = profile("qwen3.8-max");
    let context = wire_api::context(&profile, "qwen3.8-max", &RequestOptions::default());
    let item = json!({
        "type":"web_extractor_call",
        "goal":"Read the specified page and extract its key points",
        "output":"The page describes the Responses API and hosted tools."
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
                        "x_tools":{"web_extractor":{"count":2}}
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
            .and_then(|usage| usage.web_extractor_requests),
        Some(2)
    );
}

#[test]
fn stream_preserves_extractor_output_and_usage() {
    let profile = profile("qwen3.8-max");
    let options = wire_api::EncodingOptions {
        stream: true,
        file_account_scope: None,
    };
    let context = wire_api::context(&profile, "qwen3.8-max", &options);
    let item = json!({
        "type":"web_extractor_call",
        "goal":"Read the specified page and extract its key points",
        "output":"The page describes the Responses API and hosted tools."
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
            "x_tools":{"web_extractor":{"count":1}}
        }
    });
    let frames = [
        json!({"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":item}),
        json!({"type":"response.completed","sequence_number":5,"response":completed}),
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

    assert!(events.iter().any(|event| matches!(event,
        StreamEvent::ProviderContent { value, .. } if value == &item
    )));
    assert!(events.iter().any(|event| matches!(event,
        StreamEvent::End { usage, .. }
            if usage.usage.and_then(|usage| usage.server_tool_usage)
                .and_then(|usage| usage.web_extractor_requests) == Some(1)
    )));
}
