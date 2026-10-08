//! Provider error payload facts for host-owned retry decisions.
use crate::protocol::JsonPayload;

/// A typed overload payload remains distinguishable from a bare HTTP 529.
/// Native retry acceptance uses this fact before retry-decline headers.
pub fn has_overload_payload(body: &JsonPayload) -> bool {
    body.string_at("/error/type").as_deref() == Some("overloaded_error")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(source: &str) -> JsonPayload {
        JsonPayload::parse(source).expect("test source is valid JSON")
    }

    #[test]
    fn typed_overload_fact_does_not_infer_a_payload_from_status_or_message() {
        assert!(has_overload_payload(&payload(
            r#"{"error":{"type":"overloaded_error","message":"busy"}}"#
        )));
        for body in [
            "null",
            "{}",
            r#"{"status":529}"#,
            r#"{"error":{"type":"api_error","message":"overloaded_error"}}"#,
            r#"{"error":{"type":529}}"#,
            r#"{"error":"overloaded_error"}"#,
        ] {
            assert!(!has_overload_payload(&payload(body)), "{body}");
        }
    }
}
