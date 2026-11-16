use super::*;
use crate::codecs::{openai::responses::OpenAiResponsesCodec, RequestMode, WireCodec};
use crate::protocol::{ChatRequest, JsonPayload};
use crate::providers::openai::{
    native::OpenAiHostedTool,
    types::{OpenAiToolSearchConfig, RemoteMcpConfig},
};

const MODEL: &str = "gpt-6.1-sol";
const ORDINARY_SCHEMA: &str = r#"{"type":"object", "properties":{"path":{"type":"string","description":"before\ud800after"}},"required":["path"],"additionalProperties":false,"x-exact":9007199254740993}"#;

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai",
        "profile_name":"openai",
        "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_responses",
        "auth":"none",
        "extra":{"remote_mcp":"openai_responses"},
        "models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]
    }))
    .unwrap()
}

fn ordinary_tool(index: usize, deferred: bool) -> ToolSpec {
    ToolSpec {
        name: format!("read_fixture_{index}"),
        description: JsonPayload::string_from_utf16_code_units(&[0x64, 0xd800, 0x73]),
        input_schema: JsonPayload::parse(ORDINARY_SCHEMA).unwrap(),
        strict: true,
        defer_loading: deferred,
        native_options: Vec::new(),
        tool_type: None,
        extra: JsonPayload::default(),
    }
}

fn request_with_tools(count: usize) -> ChatRequest {
    let mut req = ChatRequest::new(MODEL);
    req.messages
        .push(ConversationMessage::user_text("Find a fixture tool."));
    req.tools = (0..count)
        .map(|index| ordinary_tool(index, index != 0))
        .collect();
    req
}

fn add_remote_mcp(req: &mut ChatRequest) {
    req.hosted_tools.push(
        OpenAiHostedTool::RemoteMcp(
            RemoteMcpConfig::new("fixture", "https://mcp.example.test/mcp").unwrap(),
        )
        .into(),
    );
}

fn encode(req: &ChatRequest, mode: RequestMode) -> JsonPayload {
    let profile = profile();
    let context = CodecContext::new(&profile, MODEL, mode);
    let encoded = OpenAiResponsesCodec
        .encode_request(EncodeRequest::new(req), &context)
        .unwrap();
    assert_eq!(
        OpenAiResponsesCodec
            .encoded_body_len(EncodeRequest::new(req), &context)
            .unwrap(),
        encoded.body.len(),
        "length prediction must use the same source-backed tool projection"
    );
    JsonPayload::parse(std::str::from_utf8(&encoded.body).unwrap()).unwrap()
}

fn assert_ordinary_tools(body: &JsonPayload, req: &ChatRequest) {
    for (index, tool) in req.tools.iter().enumerate() {
        let prefix = format!("/tools/{index}");
        assert_eq!(
            body.string_at(&format!("{prefix}/type")).as_deref(),
            Some("function")
        );
        assert_eq!(
            body.string_at(&format!("{prefix}/name")).as_deref(),
            Some(tool.name.as_str())
        );
        assert_eq!(
            body.raw_source_at(&format!("{prefix}/parameters"))
                .as_deref(),
            Some(tool.input_schema.as_json_source()),
            "ordinary function schemas must keep their original JSON source"
        );
        assert_eq!(
            body.payload_at(&format!("{prefix}/description"))
                .unwrap()
                .root_string_code_units(),
            Some(vec![0x64, 0xd800, 0x73])
        );
        assert_eq!(
            body.raw_source_at(&format!("{prefix}/defer_loading"))
                .as_deref(),
            tool.defer_loading.then_some("true")
        );
        assert_eq!(
            body.raw_source_at(&format!("{prefix}/strict")).as_deref(),
            Some("true")
        );
    }
}

#[test]
fn client_tool_search_keeps_exact_parameters_with_eager_and_deferred_functions_in_both_modes() {
    // Materializing either schema as a serde_json::Value would reject the lone
    // surrogate and this depth, so equality also checks the raw overlay path.
    let nested = format!("{}0{}", "[".repeat(16_384), "]".repeat(16_384));
    let parameters = JsonPayload::parse(&format!(
        r#"{{"type":"object", "properties":{{"query":{{"type":"string","description":"q\udfffz"}}}},"required":["query"],"additionalProperties":false,"x-deep":{nested},"\ud800":"\ud800"}}"#
    ))
    .unwrap();

    for ordinary_count in [0, 1, 2] {
        let mut req = request_with_tools(ordinary_count);
        req.hosted_tools.push(
            OpenAiHostedTool::ToolSearch(OpenAiToolSearchConfig {
                execution: OpenAiToolSearchExecution::Client,
                description: Some("Find fixture tools".into()),
                parameters: Some(parameters.clone()),
            })
            .into(),
        );
        add_remote_mcp(&mut req);

        let mut complete_tools = None;
        for mode in [RequestMode::Complete, RequestMode::Stream] {
            let body = encode(&req, mode);
            assert_eq!(body.array_len("/tools"), Some(ordinary_count + 2));
            assert_ordinary_tools(&body, &req);
            let prefix = format!("/tools/{ordinary_count}");
            assert_eq!(
                body.string_at(&format!("{prefix}/type")).as_deref(),
                Some("tool_search")
            );
            assert_eq!(
                body.string_at(&format!("{prefix}/execution")).as_deref(),
                Some("client")
            );
            assert_eq!(
                body.raw_source_at(&format!("{prefix}/parameters")).as_deref(),
                Some(parameters.as_json_source()),
                "client ToolSearch parameters must survive the ordinary tools overlay ({mode:?}, {ordinary_count} functions)"
            );
            assert_eq!(
                body.payload_at(&format!("{prefix}/parameters/properties/query/description"))
                    .unwrap()
                    .root_string_code_units(),
                Some(vec![0x71, 0xdfff, 0x7a])
            );
            assert_eq!(
                body.string_at(&format!("/tools/{}/type", ordinary_count + 1))
                    .as_deref(),
                Some("mcp")
            );
            assert_eq!(
                body.string_at(&format!("/tools/{}/server_label", ordinary_count + 1))
                    .as_deref(),
                Some("fixture")
            );
            let tools = body.raw_source_at("/tools").unwrap();
            if let Some(expected) = &complete_tools {
                assert_eq!(
                    &tools, expected,
                    "stream mode must not change tool definitions"
                );
            } else {
                complete_tools = Some(tools);
            }
        }
    }
}

#[test]
fn server_tool_search_and_ordinary_only_requests_keep_their_existing_wire_in_both_modes() {
    for mode in [RequestMode::Complete, RequestMode::Stream] {
        let ordinary = request_with_tools(1);
        let ordinary_body = encode(&ordinary, mode);
        assert_eq!(ordinary_body.array_len("/tools"), Some(1));
        assert_ordinary_tools(&ordinary_body, &ordinary);

        let mut server = request_with_tools(2);
        server
            .hosted_tools
            .push(OpenAiHostedTool::ToolSearch(OpenAiToolSearchConfig::default()).into());
        add_remote_mcp(&mut server);
        let server_body = encode(&server, mode);
        assert_eq!(server_body.array_len("/tools"), Some(4));
        assert_ordinary_tools(&server_body, &server);
        assert_eq!(
            server_body.raw_source_at("/tools/2").as_deref(),
            Some(r#"{"type":"tool_search"}"#),
            "server search must not receive a client parameters overlay"
        );
        assert_eq!(
            server_body.string_at("/tools/3/type").as_deref(),
            Some("mcp")
        );
    }
}
