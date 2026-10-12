use lingxi_llm_client::protocol::JsonPayload;
use lingxi_llm_client::providers::anthropic::request_policy::AnthropicRequestPolicy;
use serde_json::json;
use std::collections::BTreeMap;
#[test]
fn explicit_empty_tool_prefix_keeps_extra_body_precedence() {
    for (override_value, expected) in [
        (None, json!([])),
        (Some(json!([{"name":"parent"}])), json!([{"name":"parent"}])),
    ] {
        let mut body = JsonPayload::from_generated_value(
            &json!({"model":"claude-sonnet-4-6","messages":[],"max_tokens":64}),
        )
        .unwrap();
        let mut policy = AnthropicRequestPolicy {
            explicit_empty_tools: true,
            ..Default::default()
        };
        if let Some(value) = override_value {
            policy.extra_body = JsonPayload::from_generated_value(&json!({"tools":value})).unwrap();
        }
        policy
            .apply(
                &mut body,
                &mut BTreeMap::new(),
                &mut BTreeMap::new(),
                &mut "https://api.anthropic.com/v1/messages".into(),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body.raw_source_at("/tools").unwrap())
                .unwrap(),
            expected
        );
    }
}

#[test]
fn plain_string_content_preserves_utf16_and_sidecar_after_main_merge() {
    let mut body = JsonPayload::parse(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"\ud800"}]}]}"#,
    )
    .unwrap();
    let mut overrides = BTreeMap::from([("/messages/0/content/0/text".into(), vec![0xd800])]);
    AnthropicRequestPolicy {
        plain_text_message_content: true,
        ..Default::default()
    }
    .apply(
        &mut body,
        &mut BTreeMap::new(),
        &mut overrides,
        &mut "https://api.anthropic.com/v1/messages".into(),
    )
    .unwrap();
    assert_eq!(
        body.raw_source_at("/messages/0/content").as_deref(),
        Some(r#""\ud800""#)
    );
    assert_eq!(overrides.get("/messages/0/content"), Some(&vec![0xd800]));
    assert!(!overrides.contains_key("/messages/0/content/0/text"));
}
