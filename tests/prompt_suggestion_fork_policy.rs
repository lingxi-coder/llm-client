use lingxi_llm_client::protocol::{
    CachePosition, ChatRequest, ConversationMessage, ProtocolFamily,
};
use lingxi_llm_client::providers::{anthropic::system_prompt, response_headers};
use serde_json::{json, Value};

#[test]
fn skip_cache_write_matches_native_marker_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/prompt_suggestion_query_2_1_296.json"
    ))
    .unwrap();
    for case in fixture["cacheCases"].as_array().unwrap() {
        let mut request = ChatRequest::new("claude-sonnet-4-6");
        for (index, role) in case["roles"].as_array().unwrap().iter().enumerate() {
            let content: Vec<Value> = case["blocks"][index].as_array().unwrap().iter().map(|block| match block.as_str().unwrap() {
                "text" => json!({"type":"text","text":"x"}),
                "empty_text" => json!({"type":"text","text":""}),
                "whitespace_text" => json!({"type":"text","text":" "}),
                "thinking" => json!({"type":"thinking","text":"x","signature":null}),
                "tool_use" => json!({"type":"tool_use","id":"tool_1","name":"Read","input":{}}),
                "tool_result" => json!({"type":"tool_result","tool_use_id":"tool_1","content":"No tools needed for suggestion","is_error":true}),
                other => panic!("unknown fixture block {other}"),
            }).collect();
            request.messages.push(
                serde_json::from_value::<ConversationMessage>(
                    json!({"role":role,"content":content}),
                )
                .unwrap(),
            );
        }
        system_prompt::apply_last_message_breakpoint(&mut request, true);
        system_prompt::apply_fork_message_breakpoints(
            &mut request,
            true,
            system_prompt::ForkMessageCachePolicy {
                fork_point: case["parent"].as_u64().map(|value| value as usize),
                pin_parent: case["pin"].as_bool().unwrap_or(false),
                step_back: case["stepBack"].as_bool().unwrap_or(false),
            },
        );
        let indices: Vec<_> = request
            .prompt_cache
            .breakpoints
            .iter()
            .filter_map(|point| match point.position {
                CachePosition::Message { index, .. } => Some(index),
                _ => None,
            })
            .collect();
        assert_eq!(json!(indices), case["markerIndices"], "{case}");
    }
}

#[test]
fn fork_request_id_is_http_local_and_uses_provider_precedence() {
    let headers = vec![
        ("Request-Id".into(), "req_anthropic".into()),
        ("x-request-id".into(), "req_openai".into()),
        ("set-cookie".into(), "private=value".into()),
    ];
    let anthropic = response_headers::stream_metadata_for_route(
        ProtocolFamily::AnthropicMessages,
        "anthropic",
        &headers,
    );
    assert_eq!(anthropic["llm_client"]["http_request_id"], "req_anthropic");
    let openai =
        response_headers::stream_metadata_for_route(ProtocolFamily::OpenAiChat, "openai", &headers);
    assert_eq!(openai["llm_client"]["http_request_id"], "req_openai");
    assert!(anthropic.get("set-cookie").is_none());
    assert!(response_headers::stream_metadata_for_route(
        ProtocolFamily::AnthropicMessages,
        "anthropic",
        &[]
    )
    .is_null());
}

#[test]
fn fork_pin_provider_beta_and_taint_gate_matches_native() {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/prompt_suggestion_query_2_1_296.json"
    ))
    .unwrap();
    for case in fixture["pinGateCases"].as_array().unwrap() {
        let native = case["provider"].as_str().unwrap();
        let provider = if native == "firstParty" {
            "anthropic"
        } else {
            native
        };
        let profile=serde_json::from_value(json!({"provider_id":provider,"profile_name":"selected","protocol":"anthropic_messages","base_url":"https://selected.invalid","auth":"none","models":[]})).unwrap();
        let policy = system_prompt::ForkMessageCachePolicy::for_profile(
            &profile,
            system_prompt::GatePolicy {
                experimental_betas_disabled: case["disabled"].as_bool().unwrap(),
                hipaa_tainted: case["hipaa"].as_bool().unwrap(),
                ..Default::default()
            },
            Some(1),
            true,
            true,
        );
        assert_eq!(
            policy.pin_parent,
            case["expected"].as_bool().unwrap(),
            "{case}"
        );
        assert_eq!(policy.step_back, policy.pin_parent);
    }
}
