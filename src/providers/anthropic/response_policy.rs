//! Provider error payload facts for host-owned retry decisions.
use serde_json::Value;

/// A typed overload payload remains distinguishable from a bare HTTP 529.
/// Native retry acceptance uses this fact before retry-decline headers.
pub fn has_overload_payload(body: &Value) -> bool {
    body.get("error")
        .and_then(|error| error.get("type"))
        .and_then(Value::as_str)
        == Some("overloaded_error")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn typed_overload_fact_does_not_infer_a_payload_from_status_or_message() {
        assert!(has_overload_payload(
            &json!({"error":{"type":"overloaded_error","message":"busy"}})
        ));
        for body in [
            Value::Null,
            json!({}),
            json!({"status":529}),
            json!({"error":{"type":"api_error","message":"overloaded_error"}}),
            json!({"error":{"type":529}}),
            json!({"error":"overloaded_error"}),
        ] {
            assert!(!has_overload_payload(&body), "{body}");
        }
    }
}
