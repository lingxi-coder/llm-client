use lingxi_llm_client::providers::anthropic::request_policy::AnthropicRequestPolicy;
use serde_json::json;
use std::collections::BTreeMap;
#[test]
fn explicit_empty_tool_prefix_keeps_extra_body_precedence() {
    for (override_value, expected) in [(None,json!([])),(Some(json!([{"name":"parent"}])),json!([{"name":"parent"}]))] {
        let mut body=json!({"model":"claude-sonnet-4-6","messages":[],"max_tokens":64});
        let mut policy=AnthropicRequestPolicy { explicit_empty_tools:true,..Default::default() };
        if let Some(value)=override_value { policy.extra_body.insert("tools".into(),value); }
        policy.apply(&mut body,&mut BTreeMap::new(),&mut BTreeMap::new(),&mut "https://api.anthropic.com/v1/messages".into()).unwrap();
        assert_eq!(body["tools"],expected);
    }
}
