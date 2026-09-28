use async_trait::async_trait;
use lingxi_llm_client::providers::openai::types::{
    OpenAiToolSearchConfig, OpenAiToolSearchExecution, RemoteMcpConfig,
};
use lingxi_llm_client::{
    codecs::{openai::responses::OpenAiResponsesCodec, EncodeRequest, WireCodec},
    protocol::{
        ChatRequest, ConnectionSpec, ContentBlock, ConversationMessage, FailoverTriggers,
        HostedTool, LlmError, ProtocolFamily, ProviderProfile, Region, StopReason, StreamEvent,
        ToolSpec,
    },
    transport::{HttpRequest, StreamResponse, Transport},
    CodecContext, HttpResponse, LlmClientBuilder, RequestMode, RequestOptions,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[path = "support/wire_api.rs"]
mod wire_api;

const OPENAI_BASE: &str = "https://api.openai.com/v1";

fn profile(model: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai",
        "profile_name":"openai",
        "base_url":OPENAI_BASE,
        "protocol":"open_ai_responses",
        "auth":"none",
        "extra":{"remote_mcp":"openai_responses"},
        "models":[{"display_model":model,"request_model":model,"billing_model":model}]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"gpt-6-astra",
        "messages":[{"role":"user","content":[{"type":"text","text":"Find a tool."}]}]
    }))
    .unwrap()
}

fn deferred_tool(name: &str) -> ToolSpec {
    ToolSpec {
        tool_type: None,
        extra: serde_json::Value::Null,
        name: name.into(),
        description: format!("Look up {name}"),
        input_schema: json!({
            "type":"object",
            "properties":{"query":{"type":"string"}},
            "required":["query"],
            "additionalProperties":false
        }),
        strict: true,
        defer_loading: true,
        native_options: Vec::new(),
    }
}

fn server_search() -> HostedTool {
    lingxi_llm_client::providers::openai::native::OpenAiHostedTool::ToolSearch(
        OpenAiToolSearchConfig::default(),
    )
    .into()
}

fn encode(request: &ChatRequest, model: &str) -> Result<Value, LlmError> {
    let profile = profile(model);
    let context = CodecContext::new(&profile, model, RequestMode::Complete);
    let encoded = OpenAiResponsesCodec.encode_request(EncodeRequest::new(request), &context)?;
    Ok(serde_json::from_slice(&encoded.body).unwrap())
}

fn decode(body: Value, model: &str) -> lingxi_llm_client::protocol::ChatResponse {
    let profile = profile(model);
    let context = CodecContext::new(&profile, model, RequestMode::Complete);
    OpenAiResponsesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            },
            &context,
        )
        .unwrap()
}

#[test]
fn hosted_search_encodes_deferred_functions_and_server_mcp() {
    let mut req = request();
    req.hosted_tools.push(server_search());
    req.tools.push(deferred_tool("lookup_order"));
    let wire = encode(&req, "gpt-5.4").unwrap();
    assert_eq!(wire["tools"][0]["defer_loading"], true);
    assert_eq!(wire["tools"][1], json!({"type":"tool_search"}));

    let mut mcp_request = request();
    mcp_request.hosted_tools.push(server_search());
    mcp_request.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(
            RemoteMcpConfig::new("orders", "https://mcp.example.test/mcp")
                .unwrap()
                .with_defer_loading(true),
        )
        .into(),
    );
    let wire = encode(&mcp_request, "gpt-6-astra").unwrap();
    assert_eq!(wire["tools"][0]["type"], "tool_search");
    assert_eq!(wire["tools"][1]["type"], "mcp");
    assert_eq!(wire["tools"][1]["defer_loading"], true);
}

#[test]
fn client_search_encodes_its_schema_and_requires_gpt_5_4_or_later() {
    let mut req = request();
    req.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::ToolSearch(
            OpenAiToolSearchConfig {
                execution: OpenAiToolSearchExecution::Client,
                description: Some("Find the tools needed for this task".into()),
                parameters: Some(json!({
                    "type":"object",
                    "properties":{"goal":{"type":"string"}},
                    "required":["goal"],
                    "additionalProperties":false
                })),
            },
        )
        .into(),
    );
    assert_eq!(
        encode(&req, "gpt-5.4-mini").unwrap()["tools"][0],
        json!({
            "type":"tool_search",
            "execution":"client",
            "description":"Find the tools needed for this task",
            "parameters":{
                "type":"object",
                "properties":{"goal":{"type":"string"}},
                "required":["goal"],
                "additionalProperties":false
            }
        })
    );
    for unsupported in ["gpt-5.3", "gpt-5.4preview", "o4-mini"] {
        assert!(matches!(
            encode(&req, unsupported),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
    assert!(encode(&req, "gpt-5.5").is_ok());
    assert!(encode(&req, "gpt-6-sol").is_ok());
}

#[test]
fn client_search_calls_require_host_action_and_keep_call_id_correlated_on_replay() {
    let call = json!({
        "type":"tool_search_call",
        "execution":"client",
        "call_id":"tsc_123",
        "status":"completed",
        "arguments":{"goal":"find shipping ETA"}
    });
    let output = json!({
        "type":"tool_search_output",
        "execution":"client",
        "call_id":"tsc_123",
        "status":"completed",
        "tools":[{
            "type":"function",
            "name":"get_shipping_eta",
            "description":"Look up delivery timing",
            "defer_loading":true,
            "parameters":{"type":"object","properties":{"order_id":{"type":"string"}}}
        }]
    });
    let response = decode(
        json!({
            "id":"resp-search",
            "model":"gpt-6-astra",
            "status":"completed",
            "output":[call.clone()]
        }),
        "gpt-6-astra",
    );
    assert_eq!(
        response.stop_reason,
        StopReason::Other("requires_action".into())
    );
    assert_eq!(response.message.tool_uses().count(), 0);
    assert!(response.message.content.iter().any(|block| matches!(
        block,
        ContentBlock::ProviderContent { protocol: ProtocolFamily::OpenAiResponses, value }
            if value == &call && value["call_id"] == "tsc_123"
    )));

    let mut replay = request();
    replay.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::ToolSearch(
            OpenAiToolSearchConfig {
                execution: OpenAiToolSearchExecution::Client,
                description: Some("Find the right tool".into()),
                parameters: Some(json!({"type":"object","properties":{}})),
            },
        )
        .into(),
    );
    replay.messages = vec![ConversationMessage::assistant(vec![
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: call.clone(),
        },
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: output.clone(),
        },
    ])];
    assert_eq!(
        encode(&replay, "gpt-6-astra").unwrap()["input"],
        json!([call, output])
    );
}

#[test]
fn hosted_server_search_output_stays_native_without_becoming_a_client_tool_call() {
    let call = json!({
        "type":"tool_search_call",
        "execution":"server",
        "call_id":null,
        "status":"completed",
        "arguments":{"paths":["crm"]}
    });
    let output = json!({
        "type":"tool_search_output",
        "execution":"server",
        "call_id":null,
        "status":"completed",
        "tools":[]
    });
    let response = decode(
        json!({
            "model":"gpt-6-astra",
            "status":"completed",
            "output":[call.clone(), output.clone()]
        }),
        "gpt-6-astra",
    );
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.message.tool_uses().count(), 0);
    let native = response
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ProviderContent { value, .. } => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(native, vec![&call, &output]);
}

#[test]
fn streamed_client_search_is_native_and_ends_with_requires_action() {
    let profile = profile("gpt-6-astra");
    let context = CodecContext::new(&profile, "gpt-6-astra", RequestMode::Stream);
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&context);
    let call = json!({
        "type":"tool_search_call",
        "execution":"client",
        "call_id":"tsc_stream",
        "status":"completed",
        "arguments":{"goal":"find a function"}
    });
    let frames = [
        json!({"type":"response.created","response":{"id":"resp-stream","model":"gpt-6-astra"}}),
        json!({"type":"response.output_item.done","output_index":0,"item":call.clone()}),
        json!({"type":"response.completed","response":{"id":"resp-stream","model":"gpt-6-astra","status":"completed","output":[call.clone()]}}),
    ];
    let mut events = Vec::new();
    for frame in frames {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ProviderContent { value, .. } if value == &call))
            .count(),
        1
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::Other(reason),
            ..
        }) if reason == "requires_action"
    ));
}

struct FailingTransport(AtomicUsize);

#[async_trait]
impl Transport for FailingTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::Transport {
            message: "simulated interruption after hosted search dispatch".into(),
        })
    }
}

#[tokio::test]
async fn hosted_search_is_not_retried_on_a_fallback_connection() {
    let make_connection = |name: &str, order| {
        let mut profile = profile("gpt-6-astra");
        profile.profile_name = name.into();
        profile.connection = ConnectionSpec {
            group: Some("openai-tool-search".into()),
            connection_id: Some(name.into()),
            order,
            hidden: false,
            failover: FailoverTriggers {
                network: true,
                server_error: true,
                ..Default::default()
            },
        };
        profile
    };
    let profiles = [
        make_connection("primary", 0),
        make_connection("secondary", 1),
    ];
    let transport = Arc::new(FailingTransport(AtomicUsize::new(0)));
    let client = LlmClientBuilder::with_transport(transport.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut req = request();
    req.hosted_tools.push(server_search());
    req.tools.push(deferred_tool("lookup_order"));

    assert!(matches!(
        client
            .chat()
            .complete(&req, &RequestOptions::default())
            .await,
        Err(LlmError::Transport { .. })
    ));
    assert_eq!(transport.0.load(Ordering::SeqCst), 1);
}
