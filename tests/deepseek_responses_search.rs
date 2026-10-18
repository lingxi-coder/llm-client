//! DeepSeek Responses hosted search is model-specific and preserves native calls.
#[path = "support/wire_api.rs"]
mod wire_api;
use lingxi_llm_client::protocol::*;
use lingxi_llm_client::{
    EncodeRequest, HttpResponse, OpenAiResponsesCodec, RequestOptions, WireCodec,
};
use serde_json::{json, Value};
fn profile() -> ProviderProfile {
    lingxi_llm_client::builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "deepseek-responses")
        .unwrap()
}
fn request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({"model":model,"messages":[{"role":"user","content":[{"type":"text","text":"latest news"}]}],"hosted_tools":[{"type":"web_search","config":{}}]})).unwrap()
}
fn encode(req: &ChatRequest, model: &str) -> Result<Value, LlmError> {
    OpenAiResponsesCodec
        .encode_request(
            EncodeRequest::new(req),
            &wire_api::context(&profile(), model, &RequestOptions::default()),
        )
        .map(|r| serde_json::from_slice(&r.body).unwrap())
}
#[test]
fn only_pro_supports_hosted_search_with_actual_wire_model() {
    let body = encode(&request("deepseek/deepseek-v4-pro"), "deepseek-v4-pro").unwrap();
    assert_eq!(body["tools"], json!([{"type":"web_search"}]));
    assert!(body.get("include").is_none(), "DeepSeek ignores include");
    assert!(matches!(
        encode(&request("deepseek-flash"), "deepseek-flash"),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert!(
        lingxi_llm_client::hosted_search::supports_model_with_config(
            &profile(),
            "deepseek-v4-pro",
            &Default::default()
        )
    );
    assert!(
        !lingxi_llm_client::hosted_search::supports_model_with_config(
            &profile(),
            "deepseek-flash",
            &Default::default()
        )
    );
    assert!(lingxi_llm_client::hosted_search::supports(&profile()));
    let mut no_search = request("deepseek-flash");
    no_search.set_hosted_web_search(None);
    assert!(encode(&no_search, "deepseek-flash").is_ok());
}
#[test]
fn unsupported_search_options_are_rejected() {
    for config in [
        WebSearchConfig {
            max_uses: Some(1),
            ..Default::default()
        },
        WebSearchConfig {
            allowed_domains: vec!["example.com".into()],
            ..Default::default()
        },
        WebSearchConfig {
            blocked_domains: vec!["example.com".into()],
            ..Default::default()
        },
    ] {
        let mut req = request("deepseek-v4-pro");
        req.set_hosted_web_search(Some(config));
        assert!(encode(&req, "deepseek-v4-pro").is_err());
    }
    let mut req = request("deepseek-v4-pro");
    req.tool_choice = ToolChoice::Any;
    assert!(encode(&req, "deepseek-v4-pro").is_err());
}
fn call() -> Value {
    json!({"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","queries":["news"],"sources":[{"url":"https://example.com","title":"Source"}]}})
}
fn output() -> Value {
    json!([call(),{"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"News","annotations":[{"type":"url_citation","url":"https://example.com","title":"Source","start_index":0,"end_index":4}]}]}])
}
#[test]
fn native_search_calls_and_citations_decode_and_replay() {
    let ctx = wire_api::context(&profile(), "deepseek-v4-pro", &RequestOptions::default());
    let response = OpenAiResponsesCodec.decode_response(&HttpResponse {status:200, headers:vec![],body:json!({"id":"r1","model":"deepseek-v4-pro","status":"completed","output":output(),"usage":{"input_tokens":3,"output_tokens":4}}).to_string().into()}, &ctx).unwrap();
    assert_eq!(
        response.web_search.as_ref().unwrap().citations[0].url,
        "https://example.com"
    );
    let mut req = request("deepseek-v4-pro");
    req.messages.push(response.message);
    let body = encode(&req, "deepseek-v4-pro").unwrap();
    assert!(body["input"].as_array().unwrap().contains(&call()));
}
#[test]
fn streamed_search_calls_are_preserved_and_not_exposed_as_local_tools() {
    let ctx = wire_api::context(
        &profile(),
        "deepseek-v4-pro",
        &wire_api::EncodingOptions {
            stream: true,
            ..Default::default()
        },
    );
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&ctx);
    let frames = [
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"in_progress"}}),
        json!({"type":"response.web_search_call.searching","item_id":"ws_1","output_index":0}),
        json!({"type":"response.output_item.done","output_index":0,"item":call()}),
        json!({"type":"response.completed","response":{"id":"r1","model":"deepseek-v4-pro","status":"completed","output":output(),"usage":{"input_tokens":3,"output_tokens":4}}}),
    ];
    let events = frames
        .iter()
        .flat_map(|frame| {
            wire_api::decode_frame(decoder.as_mut(), frame.to_string().as_bytes()).unwrap()
        })
        .collect::<Vec<_>>();
    assert!(events
        .iter()
        .any(|event| matches!(event, StreamEvent::WebSearch { .. })));
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::End {
            stop_reason: StopReason::EndTurn,
            ..
        }
    )));
}
