use lingxi_llm_client::protocol::JsonPayload;
use serde_json::Value;

/// Explicitly import a bounded Value created by a test fixture into the SDK payload carrier.
pub fn generated(value: &Value) -> JsonPayload {
    JsonPayload::from_generated_value(value).expect("generated JSON fixture is valid")
}

/// Inspect an SDK source payload in a test without adding a Value conversion to production APIs.
pub fn source_matches(payload: &JsonPayload, expected: &Value) -> bool {
    serde_json::from_str::<Value>(payload.as_json_source()).is_ok_and(|actual| actual == *expected)
}

/// Read a JSON string at a pointer from the retained source for a semantic fixture assertion.
pub fn source_string_at(payload: &JsonPayload, pointer: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(payload.as_json_source()).ok()?;
    value.pointer(pointer)?.as_str().map(str::to_owned)
}

/// Read a JSON subtree from the retained source for a semantic fixture assertion.
pub fn source_value_at(payload: &JsonPayload, pointer: &str) -> Option<Value> {
    let value = serde_json::from_str::<Value>(payload.as_json_source()).ok()?;
    value.pointer(pointer).cloned()
}
