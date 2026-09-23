use lingxi_llm_client::protocol::{CapabilitySupport, ModelCapability, ModelProfile};
use serde_json::{json, Value};

fn profile(value: Value) -> ModelProfile {
    serde_json::from_value(value).expect("model profile should deserialize")
}

fn capabilities(tools: bool, reasoning: bool, vision: bool) -> Value {
    json!({
        "vision": vision,
        "documents": false,
        "tools": tools,
        "reasoning": reasoning,
        "signed_reasoning": false,
        "streaming": false,
        "structured_output": false
    })
}

fn model_json(capabilities: Value) -> Value {
    json!({
        "display_model": "model",
        "request_model": "model",
        "billing_model": "model",
        "capabilities": capabilities
    })
}

#[test]
fn legacy_true_is_supported_while_legacy_false_remains_unknown() {
    let mut value = model_json(capabilities(true, false, false));
    value["capabilities"]["streaming"] = json!(true);
    let model = profile(value);

    assert_eq!(
        model.capability_support_for(ModelCapability::Tools),
        CapabilitySupport::Supported
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Reasoning),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Vision),
        CapabilitySupport::Unknown
    );

    let serialized = serde_json::to_value(model).unwrap();
    assert!(serialized.get("capability_support").is_none());
}

#[test]
fn explicit_statuses_round_trip_and_partial_unknown_preserves_legacy_positive() {
    let mut value = model_json(capabilities(true, true, true));
    value["capability_support"] = json!({
        "tools": "unsupported",
        "vision": "unknown",
        "structured_output": "supported"
    });
    let model = profile(value);

    assert_eq!(
        model.capability_support_for(ModelCapability::Tools),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Vision),
        CapabilitySupport::Supported,
        "unknown metadata falls back to positive legacy evidence"
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Reasoning),
        CapabilitySupport::Supported,
        "omitted metadata preserves another legacy positive"
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::StructuredOutput),
        CapabilitySupport::Supported
    );

    let serialized = serde_json::to_value(&model).unwrap();
    assert_eq!(
        serialized["capability_support"],
        json!({
            "tools": "unsupported",
            "structured_output": "supported"
        })
    );
    let decoded: ModelProfile = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        decoded.capability_support_for(ModelCapability::Tools),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        decoded.capability_support_for(ModelCapability::Vision),
        CapabilitySupport::Supported
    );
}

#[test]
fn omitted_directory_facts_do_not_become_negative_capabilities() {
    let model = profile(json!({
        "display_model": "synced",
        "request_model": "synced",
        "billing_model": "synced"
    }));

    assert_eq!(
        model.capability_support_for(ModelCapability::Tools),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Vision),
        CapabilitySupport::Unknown
    );
}

#[test]
fn explicit_negative_is_distinct_from_an_omitted_fact() {
    let mut value = model_json(capabilities(false, false, false));
    value["capability_support"] = json!({ "tools": "unsupported" });
    let model = profile(value);

    assert_eq!(
        model.capability_support_for(ModelCapability::Tools),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Reasoning),
        CapabilitySupport::Unknown
    );
}
