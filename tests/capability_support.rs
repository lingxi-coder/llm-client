use lingxi_llm_client::protocol::{CapabilitySupport, ModelCapability, ModelProfile};
use serde_json::{json, Value};

fn model_json() -> Value {
    json!({"display_model":"model","request_model":"model","billing_model":"model"})
}

#[test]
fn explicit_statuses_round_trip_without_boolean_fallbacks() {
    let mut value = model_json();
    value["capability_support"] =
        json!({"tools":"unsupported","vision":"unknown","structured_output":"supported"});
    let model: ModelProfile = serde_json::from_value(value).unwrap();
    assert_eq!(
        model.capability_support_for(ModelCapability::Tools),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Vision),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::Reasoning),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        model.capability_support_for(ModelCapability::StructuredOutput),
        CapabilitySupport::Supported
    );
    let serialized = serde_json::to_value(&model).unwrap();
    assert_eq!(
        serialized["capability_support"],
        json!({"tools":"unsupported","structured_output":"supported"})
    );
    assert_eq!(
        serde_json::from_value::<ModelProfile>(serialized).unwrap(),
        model
    );
}

#[test]
fn omitted_directory_facts_stay_unknown() {
    let model: ModelProfile = serde_json::from_value(model_json()).unwrap();
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
fn removed_boolean_capability_schema_is_rejected() {
    let mut value = model_json();
    value["capabilities"] = json!({"tools":true});
    assert!(serde_json::from_value::<ModelProfile>(value).is_err());
}
