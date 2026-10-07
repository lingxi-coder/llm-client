//! Native refusal-state request headers. State acquisition belongs to the host.
use super::request_policy::set_header;

/// Native `efn`: one visible ASCII character, then at most 254 ASCII spaces or
/// visible characters. The value is retained verbatim, including trailing spaces.
pub fn origin_header(value: &str) -> Option<&str> {
    let bytes = value.as_bytes();
    (matches!(bytes.len(), 1..=255)
        && matches!(bytes[0], 0x21..=0x7e)
        && bytes[1..].iter().all(|byte| matches!(byte, 0x20..=0x7e)))
    .then_some(value)
}

pub fn apply_request_headers(
    headers: &mut std::collections::BTreeMap<String, String>,
    first_party: bool,
    armed: bool,
    refusal_occurred: bool,
    lane_enabled: bool,
    origin_request_id: Option<&str>,
) {
    if !first_party || !(armed || refusal_occurred && lane_enabled) {
        return;
    }
    set_header(headers, "x-is-refusal-fallback", "true");
    if let Some(origin) = origin_request_id.and_then(origin_header) {
        // Native Headers.set strips HTTP whitespace after efn validation.
        set_header(
            headers,
            "x-cc-fallback-latched-by",
            origin.trim_end_matches([' ', '\t', '\r', '\n']),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    #[test]
    fn native_header_gate_origin_validation_and_headers_normalization_match() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/refusal_headers_2_1_288.json"
        ))
        .unwrap();
        for case in fixture["headerCases"].as_array().unwrap() {
            let origin = case["id"].as_str();
            assert_eq!(
                json!(origin.and_then(origin_header)),
                case["validated"],
                "{case}"
            );
            let mut headers: std::collections::BTreeMap<String, String> = [
                ("x-is-refusal-fallback".into(), "explicit".into()),
                ("x-cc-fallback-latched-by".into(), "prior".into()),
            ]
            .into_iter()
            .collect();
            apply_request_headers(
                &mut headers,
                case["firstParty"].as_bool().unwrap(),
                case["armed"].as_bool().unwrap(),
                case["occurred"].as_bool().unwrap(),
                case["lane"].as_bool().unwrap(),
                origin,
            );
            let actual: std::collections::BTreeMap<_, _> = headers.into_iter().collect();
            assert_eq!(json!(actual), case["expected"], "{case}");
        }
    }
}
